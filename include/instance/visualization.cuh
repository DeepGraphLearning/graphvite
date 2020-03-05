/**
 * Copyright 2019 MilaGraph. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @author Zhaocheng Zhu
 */

#pragma once

#include <thread>
#include <unordered_map>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#ifndef NO_FAISS
#include "faiss/gpu/GpuIndexFlat.h"
#include "faiss/gpu/StandardGpuResources.h"
#endif

#include "graph.cuh"
#include "core/solver.h"
#include "model/visualization.h"
#include "gpu/visualization.cuh"

namespace py = pybind11;

/**
 * @page Graph & High-dimensional Data Visualization
 *
 * Graph & high-dimensional data visualization is an instantiation of the system.
 * The visualization process can be seen as training 2d or 3d embeddings for a
 * graph, or a KNN graph built on high-dimensional vectors.
 *
 * In visualization, there is only one embedding matrix, namely the coordinates.
 * The coordinates are related to both head and tail partitions,
 * and we implement them in a way similar to knowledge graph embedding.
 *
 * Currently, our visualization solver supports LargeVis.
 *
 * Reference:
 *
 * 1) LargeVis
 * https://arxiv.org/pdf/1602.00370.pdf
 *
 */

namespace graphvite {

template<class _KNNGraph>
class KNNWorker {
public:
    typedef _KNNGraph KNNGraph;
    typedef typename KNNGraph::Index Index;

    KNNGraph *graph;
    int device_id;
#ifndef NO_FAISS
    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexFlatConfig config;
    std::shared_ptr<faiss::gpu::GpuIndexFlatL2> index;
#endif

    KNNWorker(KNNGraph *_graph, int _device_id) :
            graph(_graph), device_id(_device_id) {
#ifndef NO_FAISS
        config.device = device_id;
#endif
    }

    void build() {
#ifndef NO_FAISS
        index = std::make_shared<faiss::gpu::GpuIndexFlatL2>(&resources, graph->dim, config);
#endif
    }

    void search(Index start, Index end) {
#ifndef NO_FAISS
        index->add(graph->num_vertex, graph->vectors.data());
        // faiss returns squared l2 distances
        index->search(end - start, &graph->vectors[start], graph->num_neighbor + 1,
                      &graph->distances[start], &graph->labels[start]);
        index->reset();
#endif
    }
};

/**
 * K-nearest neighbor graphs
 *
 * Reference:
 *
 * 1) LargeVis
 * https://arxiv.org/pdf/1602.00370.pdf
 *
 * @tparam _Index integral type of node indexes
 */
template<class _Index = size_t>
class KNNGraph : public Graph<_Index> {
public:
    typedef Graph<_Index> Base;
    typedef KNNWorker<KNNGraph> KNNWorker;
    USING_GRAPH(Base);

    typedef _Index Index;

    int dim, num_neighbor;
    float perplexity;
    bool vector_normalization;
    int num_worker, num_thread;
    std::vector<float> vectors;
#ifndef NO_FAISS
    std::vector<faiss::Index::distance_t> distances;
    std::vector<faiss::Index::idx_t> labels;
#endif
    std::vector<KNNWorker *> workers;

    /**
     * @brief Construct a KNN graph
     * @param device_ids list of GPU ids, {} for auto
     * @param num_thread_per_worker number of CPU thread per GPU
     */
    KNNGraph(std::vector<int> device_ids = {}, int num_thread_per_worker = kAuto) {
        if (device_ids.empty()) {
            CUDA_CHECK(cudaGetDeviceCount(&num_worker));
            CHECK(num_worker) << "No GPU devices found";
            for (int i = 0; i < num_worker; i++)
                device_ids.push_back(i);
        }
        else
            num_worker = device_ids.size();
        if (num_thread_per_worker == kAuto)
            num_thread_per_worker = std::thread::hardware_concurrency() / num_worker;
        num_thread = num_thread_per_worker * num_worker;
        LOG_IF(WARNING, num_thread > std::thread::hardware_concurrency())
                << "#CPU threads is beyond the hardware concurrency";
#ifdef NO_FAISS
        LOG(FATAL) << "GraphVite is not compiled with FAISS enabled";
#endif
        for (auto &&device_id : device_ids)
            workers.push_back(new KNNWorker(this, device_id));
    }

    ~KNNGraph() {
        for (auto &&worker : workers)
            delete worker;
    }

    /** Clear the graph and free CPU memory */
    void clear() override {
        Base::clear();
        as_undirected = true;
        dim = 0;
    }

    inline std::string name() const override {
        std::stringstream ss;
        ss << "KNNGraph<" << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    inline std::string graph_info() const override {
        std::stringstream ss;
        ss << "#vertex: " << num_vertex << ", #nearest neighbor: " << num_neighbor << std::endl;
        ss << "perplexity: " << perplexity << ", vector normalization: " << pretty::yes_no(vector_normalization);
        return ss.str();
    }

    /** Normalize the input vectors. This function can be parallelized. */
    void normalize_vector(int start, int end) {
        for (int i = start; i < end; i++) {
            double mean = 0;
            for (Index j = i; j < num_vertex * dim; j += dim)
                mean += vectors[j];
            mean /= num_vertex;
            float max = 0;
            for (Index j = i; j < num_vertex * dim; j += dim) {
                vectors[j] -= mean;
                max = std::max(max, abs(vectors[j]));
            }
            for (Index j = i; j < num_vertex * dim; j += dim)
                vectors[j] /= (max + kEpsilon);
        }
    }

    /** Compute the weight for each edge. This function can be parallelized. */
    void compute_weight(Index start, Index end) {
#ifndef NO_FAISS
        for (Index i = start; i < end; i++) {
            vertex_edges[i].resize(num_neighbor);
            for (int j = 0; j < num_neighbor; j++) {
                int k = i * (num_neighbor + 1) + j + 1;
                vertex_edges[i][j] = std::make_tuple(labels[k], distances[k]);
            }
            float norm = 0, entropy;
            for (auto &&vertex_edge : vertex_edges[i]) {
                float weight = std::get<1>(vertex_edge);
                norm += weight;
            }
            float low = -1, high = -1, beta = 1;
            for (int j = 0; j < 100; j++) {
                norm = 0;
                entropy = 0;
                for (auto &&vertex_edge : vertex_edges[i]) {
                    float weight = std::get<1>(vertex_edge);
                    norm += std::exp(-beta * weight);
                    entropy += beta * weight * std::exp(-beta * weight);
                }
                entropy = entropy / norm + log(norm);
                if (abs(entropy - log(perplexity)) < 1e-5)
                    break;
                if (entropy > log(perplexity)) {
                    low = beta;
                    beta = high < 0 ? beta * 2 : (beta + high) / 2;
                }
                else {
                    high = beta;
                    beta = low < 0 ? beta / 2 : (beta + high) / 2;
                }
            }
            for (auto &&vertex_edge : vertex_edges[i]) {
                float &weight = std::get<1>(vertex_edge);
                weight = exp(-beta * weight) / norm;
            }
            vertex_weights[i] = 1;
        }
#endif
    }

    /** Symmetrize the graph. This function can be parallelized. */
    void symmetrize(Index start, Index end) {
        for (Index i = start; i < end; i++)
            for (auto &&edge_i : vertex_edges[i]) {
                Index j = std::get<0>(edge_i);
                if (i < j)
                    for (auto &&edge_j : vertex_edges[j])
                        if (std::get<0>(edge_j) == i) {
                            float weight = (std::get<1>(edge_i) + std::get<1>(edge_j)) / 2;
                            std::get<1>(edge_i) = weight;
                            std::get<1>(edge_j) = weight;
                            break;
                        }
            }
    }

    /** Build the KNN graph from input vectors */
    void build() {
#ifndef NO_FAISS
        num_vertex = vectors.size() / dim;
        num_edge = num_vertex * num_neighbor;

        distances.resize(num_vertex * (num_neighbor + 1));
        labels.resize(num_vertex * (num_neighbor + 1));
        vertex_edges.resize(num_vertex);
        vertex_weights.resize(num_vertex, 0);
        for (auto &&worker : workers)
            worker->build();

        std::vector<std::thread> worker_threads(num_worker);
        std::vector<std::thread> build_threads(num_thread);
        if (vector_normalization) {
            int work_load = (dim + num_thread - 1) / num_thread;
            for (int i = 0; i < num_thread; i++)
                build_threads[i] = std::thread(&KNNGraph::normalize_vector, this, work_load * i,
                                               std::min(work_load * (i + 1), dim));
            for (auto &&thread : build_threads)
                thread.join();
        }

        Index work_load = (num_vertex + num_worker - 1) / num_worker;
        for (int i = 0; i < num_worker; i++)
            worker_threads[i] = std::thread(&KNNWorker::search, workers[i], work_load * i,
                                            std::min(work_load * (i + 1), num_vertex));
        for (auto &&thread : worker_threads)
            thread.join();

        work_load = (num_vertex + num_thread - 1) / num_thread;
        for (int i = 0; i < num_thread; i++)
            build_threads[i] = std::thread(&KNNGraph::compute_weight, this, work_load * i,
                                           std::min(work_load * (i + 1), num_vertex));
        for (auto &&thread : build_threads)
            thread.join();

        for (int i = 0; i < num_thread; i++)
            build_threads[i] = std::thread(&KNNGraph::symmetrize, this, work_load * i,
                                           std::min(work_load * (i + 1), num_vertex));
        for (auto &&thread : build_threads)
            thread.join();
#endif
    }

    /**
     * @brief Build a KNN graph from a vector-list file. Store the graph in an adjacency list.
     * @param vector_file vector file
     * @param _num_neighbor number of neighbors for each node
     * @param _perplexity perplexity for the neighborhood of each node
     * @param _vector_normalization normalize the input vectors or not
     * @param delimiters string of delimiter characters
     * @param comment prefix of comment strings
     */
    void load_file(const char *vector_file, int _num_neighbor = 200, float _perplexity = 30,
                   bool _vector_normalization = true, const char *delimiters = " \t\r\n", const char *comment = "#") {
        LOG(INFO) << "loading vectors from " << vector_file;
        clear();

        num_neighbor = _num_neighbor;
        perplexity = _perplexity;
        CHECK(perplexity <= num_neighbor) << "`perplexity` should be no larger than `#neighbor`";
        vector_normalization = _vector_normalization;

        FILE *fin = fopen(vector_file, "r");
        CHECK(fin) << "File `" << vector_file << "` doesn't exist";
        fseek(fin, 0, SEEK_END);
        size_t fsize = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        char line[kMaxLineLength];

        while (fgets(line, kMaxLineLength, fin) != nullptr) {
            LOG_EVERY_N(INFO, 1e5) << 100.0 * ftell(fin) / fsize << "%";

            char *comment_str = strstr(line, comment);
            if (comment_str)
                *comment_str = 0;

            int current_dim = 0;
            for (char *f_str = strtok(line, delimiters); f_str; f_str = strtok(nullptr, delimiters)) {
                float f = atof(f_str);
                vectors.push_back(f);
                current_dim++;
            }
            if (!current_dim)
                continue;
            if (!dim)
                dim = current_dim;
            CHECK(current_dim == dim)
                    << "Incompatible #dimension. Expect " << dim << ", but " << current_dim << " is found";
        }
        fclose(fin);
        LOG(INFO) << "building KNN graph";
        build();

        LOG(WARNING) << pretty::block(info());
    }

    /**
     * @brief Build a KNN graph from a vector list. Store the graph in an adjacency list.
     * @param _vectors vector list
     * @param _num_neighbor number of neighbors for each node
     * @param _perplexity perplexity for the neighborhood of each node
     * @param _normalized_vector normalize the input vectors or not
     */
    void load_vectors(const std::vector<std::vector<float>> &_vectors, int _num_neighbor = 200, float _perplexity = 30,
                      bool _normalized_vector = true) {
        clear();

        num_neighbor = _num_neighbor;
        perplexity = _perplexity;
        CHECK(perplexity <= num_neighbor) << "`perplexity` should be no larger than `#neighbor`";
        vector_normalization = _normalized_vector;

        for (auto &&vector : _vectors) {
            int current_dim = vector.size();
            vectors.insert(vectors.end(), vector.begin(), vector.end());
            if (!dim)
                dim = current_dim;
            CHECK(current_dim == dim)
                    << "Incompatible #dimension. Expect " << dim << ", but " << current_dim << " is found";
        }
        build();

        LOG(WARNING) << pretty::block(info());
    }

    /**
     * @brief Build a KNN graph from a numpy array. Store the graph in an adjacency list.
     * @param _array 2D numpy array
     * @param _num_neighbor number of neighbors for each node
     * @param _perplexity perplexity for the neighborhood of each node
     * @param _normalized_vector normalize the input vectors or not
     */
    void load_numpy(const py::array_t<float> &_array, int _num_neighbor = 200, float _perplexity = 30,
                    bool _normalized_vector = true) {
        CHECK(_array.ndim() == 2) << "Expect a 2d array, but a " << _array.ndim() << "d array is found";
        clear();

        num_neighbor = _num_neighbor;
        perplexity = _perplexity;
        CHECK(perplexity <= num_neighbor) << "`perplexity` should be no larger than `#neighbor`";
        vector_normalization = _normalized_vector;

        auto array = _array.unchecked();
        num_vertex = array.shape(0);
        dim = array.shape(1);
        vectors.resize(array.size());
        for (Index i = 0; i < num_vertex; i++)
            for (int j = 0; j < dim; j++)
                vectors[i * dim + j] = array(i, j);
        build();

        LOG(WARNING) << pretty::block(info());
    }
};

template <size_t _dim, class _Float, class _Index>
class VisualizationSolver;

/** Edge sampler for visualization */
template <class _Solver>
class VisualizationSampler : public SamplerMixin<_Solver> {
public:
    typedef SamplerMixin<_Solver> Base;
    USING_SAMPLER_MIXIN(Base);
    using Base::Base;

    /** Return no additional attributes */
    inline Attributes get_attributes(const Edge &edge) const override {
        return Attributes();
    }
};

/** Training worker for visualization */
template <class _Solver>
class VisualizationWorker : public WorkerMixin<_Solver> {
    typedef WorkerMixin<_Solver> Base;
    USING_WORKER_MIXIN(Base);
    using Base::Base;

    typedef VisualizationSolver<Solver::dim, Float, Index> VisualizationSolver;

    /**
     * Call the corresponding GPU kernel for training
     * (LargeVis) * (SGD, Momentum, AdaGrad, RMSprop, Adam)
     */
    bool train_dispatch() override {
        using namespace gpu;
        VisualizationSolver *solver = reinterpret_cast<VisualizationSolver *>(this->solver);

        switch (num_moment) {
            case 0: {
                decltype(&visualization::train<Vector, Index, LargeVis, kSGD>) train = nullptr;
                if (solver->model == "LargeVis") {
                    if (optimizer.type == "SGD")
                        train = &visualization::train<Vector, Index, LargeVis, kSGD>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1], batch, negative_batch, loss,
                                    optimizer, solver->negative_weight
                    );
                    return true;
                }
                break;
            }
            case 1: {
                decltype(&visualization::train_1_moment<Vector, Index, LargeVis, kMomentum>) train = nullptr;
                if (solver->model == "LargeVis") {
                    if (optimizer.type == "Momentum")
                        train = &visualization::train_1_moment<Vector, Index, LargeVis, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &visualization::train_1_moment<Vector, Index, LargeVis, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &visualization::train_1_moment<Vector, Index, LargeVis, kRMSprop>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1], (*moments[0])[0], (*moments[1])[0],
                                    batch, negative_batch, loss, optimizer, solver->negative_weight
                    );
                    return true;
                }
                break;
            }
            case 2: {
                decltype(&visualization::train_2_moment<Vector, Index, LargeVis, kAdam>) train = nullptr;
                if (solver->model == "LargeVis") {
                    if (optimizer.type == "Adam")
                        train = &visualization::train_2_moment<Vector, Index, LargeVis, kAdam>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1],
                                    (*moments[0])[0], (*moments[1])[0], (*moments[0])[1], (*moments[1])[1],
                                    batch, negative_batch, loss, optimizer, solver->negative_weight
                    );
                    return true;
                }
                break;
            }
        }
        return false;
    }

    virtual bool predict_dispatch() {
        return false;
    }
};

/**
 * @brief Visualization solver
 * @tparam _dim dimension of embeddings
 * @tparam _Float floating type of parameters
 * @tparam _Index integral type of node indexes
 */
template <size_t _dim, class _Float = float, class _Index = size_t>
class VisualizationSolver :
        public SolverMixin<_dim, _Float, _Index, Graph, VisualizationSampler, VisualizationWorker> {
public:
    typedef SolverMixin<_dim, _Float, _Index, Graph, VisualizationSampler, VisualizationWorker> Base;
    USING_SOLVER_MIXIN(Base);
    using Base::Base;

    std::shared_ptr<std::vector<Vector>> coordinates;

    /**
     * @brief Return the protocols of embeddings
     *
     * Head / tail coordinates are binded to head /tail partitions respectively.
     * The two embeddings share the same underlying storage. They are updated in place.
     */
    inline std::vector<Protocol> get_protocols() const override {
        return {kHeadPartition | kInPlace, kTailPartition | kInPlace | kSharedWithPredecessor};
    };

    /** Return the protocol of negative sampling */
    inline Protocol get_sampler_protocol() const override {
        return kTailPartition;
    }

    /**
     * @brief Return the shapes of embeddings
     *
     * Shapes of both head and tail coordinates can be inferred from the graph.
     */
    inline std::vector<Index> get_shapes() const override {
        return {kAuto, kAuto};
    }

    /** Return all available models of the solver */
    inline std::set<std::string> get_available_models() const override {
        return {"LargeVis"};
    }

    /** Return the default optimizer type and its hyperparameters */
    inline Optimizer get_default_optimizer() const override {
        return Adam(0.5, 1e-5);
    }

    /** Build alias reference for embeddings */
    inline void build_alias() override {
        coordinates = embeddings[0];
    }

    /** Initialize the embeddings */
    void init_embeddings() override {
        std::uniform_real_distribution<Float> init(-5e-5 / dim, 5e-5 / dim);
        for (auto &&coordinate : *coordinates)
            for (int i = 0; i < dim; i++)
                coordinate[i] = init(seed);
    }

    inline std::string name() const override {
        std::stringstream ss;
        ss << "VisualizationSolver<" << dim << ", " << pretty::type2name<Float>()
           << ", " << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    /**
     * @brief Train visualization
     * @param _model "LargeVis"
     * @param _num_epoch number of epochs, i.e. #positive edges / |E|
     * @param _resume resume training from learned embeddings or not
     * @param _sample_batch_size batch size of samples in samplers
     * @param _positive_reuse times of reusing positive samples
     * @param _negative_sample_exponent exponent of degrees in negative sampling
     * @param _negative_weight weight for each negative sample
     * @param _log_frequency log every log_frequency batches
     */
    void train(const std::string &_model = "LargeVis", int _num_epoch = 50, bool _resume = false,
               int _sample_batch_size = 2000, int _positive_reuse = 5, float _negative_sample_exponent = 0.75,
               float _negative_weight = 3, int _log_frequency = 1000) {
        Base::train(_model, _num_epoch, _resume, _sample_batch_size, _positive_reuse, _negative_sample_exponent,
                    _negative_weight, _log_frequency);
    }
};

} // namespace graphvite