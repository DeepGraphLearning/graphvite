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

#include <unordered_set>
#include <unordered_map>

#include "core/graph.h"
#include "core/solver.h"
#include "gpu/graph.cuh"

/**
 * @page Node Embedding
 *
 * Node embedding is an instantiation of the system on normal graphs (e.g. social network, citation network)
 *
 * In node embedding, there are two embedding matrices, namely the vertex embeddings and the context embeddings.
 * During training, the samplers generate positive edges on multiple CPUs
 * and the workers generate negative context nodes on GPUs.
 * Then the workers pick vertex and context embeddings according to the samples, and update the embeddings with SGD.
 *
 * Currently, our graph solver supports DeepWalk, LINE and node2vec.
 *
 * Reference:
 *
 * 1) DeepWalk
 * https://arxiv.org/pdf/1403.6652.pdf
 *
 * 2) LINE
 * https://arxiv.org/pdf/1503.03578.pdf
 *
 * 3) node2vec
 * https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf
 */

namespace graphvite {

/**
 * @brief Normal graphs without attributes
 * @tparam _Index integral type of node indexes
 */
template<class _Index = size_t>
class Graph : public GraphMixin<_Index> {
public:
    typedef GraphMixin<_Index> Base;
    USING_GRAPH_MIXIN(Base);

    typedef _Index Index;

    std::unordered_map<std::string, Index> name2id;
    std::vector<std::string> id2name;

    bool as_undirected, normalization;

#define USING_GRAPH(type) \
    USING_GRAPH_MIXIN(type); \
    using type::name2id; \
    using type::id2name; \
    using type::as_undirected; \
    using type::normalization; \
    using type::normalize

    /** Clear the graph and free CPU memory */
    void clear() override {
        Base::clear();
        decltype(name2id)().swap(name2id);
        decltype(id2name)().swap(id2name);
    }

    inline std::string name() const override {
        std::stringstream ss;
        ss << "Graph<" << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    inline std::string graph_info() const override {
        std::stringstream ss;
        ss << "#vertex: " << num_vertex << ", #edge: " << num_edge << std::endl;
        ss << "as undirected: " << pretty::yes_no(as_undirected)
           << ", normalization: " << pretty::yes_no(normalization);
        return ss.str();
    }

    void normalize() {
        std::vector<float> context_weights(num_vertex);
        for (Index u = 0; u < num_vertex; u++)
            for (auto &&vertex_edge : vertex_edges[u]) {
                Index v = std::get<0>(vertex_edge);
                float w = std::get<1>(vertex_edge);
                context_weights[v] += w;
            }
        for (Index u = 0; u < num_vertex; u++) {
            float weight = 0;
            for (auto &&vertex_edge : vertex_edges[u]) {
                Index v = std::get<0>(vertex_edge);
                float &w = std::get<1>(vertex_edge);
                w /= sqrt(vertex_weights[u] * context_weights[v]);
                weight += w;
            }
            vertex_weights[u] = weight;
        }
    }

    /** Add an edge to the adjacency list */
    void add_edge(const std::string &u_name, const std::string &v_name, float w) {
        Index u, v;
        auto u_iter = name2id.find(u_name);
        if (u_iter != name2id.end())
            u = u_iter->second;
        else {
            u = num_vertex++;
            name2id[u_name] = u;
            id2name.push_back(u_name);
            vertex_edges.push_back(std::vector<VertexEdge>());
            vertex_weights.push_back(0);
        }
        auto v_iter = name2id.find(v_name);
        if (v_iter != name2id.end())
            v = v_iter->second;
        else {
            v = num_vertex++;
            name2id[v_name] = v;
            id2name.push_back(v_name);
            vertex_edges.push_back(std::vector<VertexEdge>());
            vertex_weights.push_back(0);
        }
        vertex_edges[u].push_back(std::make_tuple(v, w));
        vertex_weights[u] += w;
        if (as_undirected && u != v) {
            vertex_edges[v].push_back(std::make_tuple(u, w));
            vertex_weights[v] += w;
        }
        num_edge++;
    }

    /**
     * @brief Load a graph from an edge-list file. Store the graph in an adjacency list.
     * @param file_name file name
     * @param _as_undirected symmetrize the graph or not
     * @param _normalization normalize the adjacency matrix or not
     * @param delimiters string of delimiter characters
     * @param comment prefix of comment strings
     */
    void load_file(const char *file_name, bool _as_undirected = true, bool _normalization = false,
                   const char *delimiters = " \t\r\n", const char *comment = "#") {
        LOG(INFO) << "loading graph from " << file_name;
        clear();
        as_undirected = _as_undirected;
        normalization = _normalization;

        FILE *fin = fopen(file_name, "r");
        CHECK(fin) << "File `" << file_name << "` doesn't exist";
        fseek(fin, 0, SEEK_END);
        size_t fsize = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        char line[kMaxLineLength];

        for (size_t i = 1; fgets(line, kMaxLineLength, fin); i++) {
            LOG_EVERY_N(INFO, 1e7) << 100.0 * ftell(fin) / fsize << "%";

            char *comment_str = strstr(line, comment);
            if (comment_str)
                *comment_str = 0;

            char *u_name = strtok(line, delimiters);
            if (!u_name)
                continue;
            char *v_name = strtok(nullptr, delimiters);
            char *w_str = strtok(nullptr, delimiters);
            char *more = strtok(nullptr, delimiters);
            CHECK(v_name && !more) << "Invalid format at line " << i;

            float w = w_str ? atof(w_str) : 1;
            add_edge(u_name, v_name, w);
        }
        fclose(fin);
        if (normalization)
            normalize();

        LOG(WARNING) << pretty::block(info()) ;
    }

    /**
     * @brief Load a graph from an edge list. Store the graph in an adjacency list.
     * @param edge_list edge list
     * @param _as_undirected symmetrize the graph or not
     * @param _normalization normalize the adjacency matrix or not
     */
    void load(const std::vector<std::tuple<std::string, std::string, float>> &edge_list, bool _as_undirected = true,
              bool _normalization = false) {
        clear();
        as_undirected = _as_undirected;
        normalization = _normalization;

        for (auto &&edge : edge_list) {
            auto &u_name = std::get<0>(edge);
            auto &v_name = std::get<1>(edge);
            float w = std::get<2>(edge);

            add_edge(u_name, v_name, w);
        }
        if (normalization)
            normalize();

        LOG(WARNING) << pretty::block(info());
    }

    /**
     * @brief Save the graph in edge-list format
     * @param file_name file name
     * @param weighted save edge weights or not
     * @param anonymous save node names or not
     */
    void save(const char *file_name, bool weighted = true, bool anonymous = false) {
        LOG(INFO) << "Saving weighted graph to " << file_name;

        FILE *fout = fopen(file_name, "w");

        for (unsigned long long i = 0; i < num_vertex; i++)
            for (auto &&vertex_edge : vertex_edges[i]) {
                unsigned long long j = std::get<0>(vertex_edge);
                float w = std::get<1>(vertex_edge);
                if (anonymous)
                    fprintf(fout, "%llu\t%llu", i, j);
                else
                    fprintf(fout, "%s\t%s", id2name[i].c_str(), id2name[j].c_str());
                if (weighted)
                    fprintf(fout, "\t%f", w);
                fputc('\n', fout);
            }
        fclose(fout);
    }
};

template<size_t _dim, class _Float, class _Index>
class GraphSolver;

/** Edge sampler for graphs */
template<class _Solver>
class GraphSampler : public SamplerMixin<_Solver> {
public:
    typedef SamplerMixin<_Solver> Base;
    USING_SAMPLER_MIXIN(Base);
    using Base::Base;

    typedef GraphSolver<Solver::dim, Float, Index> GraphSolver;

    /** Return no additional attributes */
    inline Attributes get_attributes(const Edge &edge) const override {
        return Attributes();
    }

    /** Sample edges from biased random walks. This function can be parallelized. */
    void sample_biased_random_walk(int start, int end) {
        GraphSolver *solver = reinterpret_cast<GraphSolver *>(this->solver);

        CHECK(pool_size % solver->shuffle_base == 0)
                << "Can't perform pseudo shuffle on " << pool_size << " elements by a shuffle base of "
                << solver->shuffle_base << ". Try setting the episode size to a multiple of the shuffle base";
        CUDA_CHECK(cudaSetDevice(device_id));

        random.to_host();
        CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));

        auto &sample_pool = solver->sample_pools[solver->pool_id ^ 1];
        std::vector<std::vector<int>> offsets(solver->num_partition);
        for (auto &&partition_offsets : offsets)
            partition_offsets.resize(solver->num_partition, start);
        std::vector<std::vector<std::pair<int, Index>>> head_chains(solver->random_walk_batch_size);
        std::vector<std::vector<std::pair<int, Index>>> tail_chains(solver->random_walk_batch_size);
        for (auto &&head_chain : head_chains)
            head_chain.resize(solver->random_walk_length + 1);
        for (auto &&tail_chain : tail_chains)
            tail_chain.resize(solver->random_walk_length + 1);
        std::vector<int> sample_lengths(solver->random_walk_batch_size);
        int num_complete = 0, rand_id = 0;
        while (num_complete < solver->num_partition * solver->num_partition) {
            for (int i = 0; i < solver->random_walk_batch_size; i++) {
                if (rand_id > kRandBatchSize - solver->random_walk_length * 2) {
                    random.to_host();
                    CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));
                    rand_id = 0;
                }
                size_t edge_id = solver->edge_table.sample(random[rand_id++], random[rand_id++]);

                Index current = std::get<0>(solver->graph->edges[edge_id]);
                head_chains[i][0] = solver->head_locations[current];
                tail_chains[i][0] = solver->tail_locations[current];
                current = std::get<1>(solver->graph->edges[edge_id]);
                head_chains[i][1] = solver->head_locations[current];
                tail_chains[i][1] = solver->tail_locations[current];
                sample_lengths[i] = solver->random_walk_length;
                for (int j = 2; j <= solver->random_walk_length; j++)
                    if (!solver->graph->vertex_edges[current].empty()) {
                        Index neighbor_id = solver->edge_edge_tables[edge_id].sample(
                                random[rand_id]++, random[rand_id]++);
                        edge_id = solver->graph->flat_offsets[current] + neighbor_id;
                        current = std::get<0>(solver->graph->vertex_edges[current][neighbor_id]);
                        head_chains[i][j] = solver->head_locations[current];
                        tail_chains[i][j] = solver->tail_locations[current];
                    } else {
                        sample_lengths[i] = j - 1;
                        break;
                    }
            }
            for (int i = 0; i < solver->random_walk_batch_size; i++) {
                for (int j = 0; j < sample_lengths[i]; j++) {
                    for (int k = 1; k <= solver->augmentation_step; k++) {
                        if (j + k > sample_lengths[i])
                            break;
                        int head_partition_id = head_chains[i][j].first;
                        int tail_partition_id = tail_chains[i][j + k].first;
                        int &offset = offsets[head_partition_id][tail_partition_id];
                        if (offset < end) {
                            auto &pool = sample_pool[head_partition_id][tail_partition_id];
                            Index head_local_id = head_chains[i][j].second;
                            Index tail_local_id = tail_chains[i][j + k].second;
                            // pseudo shuffle
                            int shuffled_offset = offset % solver->shuffle_base * (pool_size / solver->shuffle_base)
                                                  + offset / solver->shuffle_base;
                            pool[shuffled_offset] = std::make_tuple(head_local_id, tail_local_id);
                            if (++offset == end)
                                num_complete++;
                        }
                    }
                }
            }
        }
    }

    /** Sample edges from random walks. This function can be parallelized. */
    void sample_random_walk(int start, int end) {
        GraphSolver *solver = reinterpret_cast<GraphSolver *>(this->solver);

        CHECK(pool_size % solver->shuffle_base == 0)
                << "Can't perform pseudo shuffle on " << pool_size << " elements by a shuffle base of "
                << solver->shuffle_base << ". Try setting the episode size to a multiple of the shuffle base";
        CUDA_CHECK(cudaSetDevice(device_id));

        random.to_host();
        CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));

        auto &sample_pool = solver->sample_pools[solver->pool_id ^ 1];
        std::vector<std::vector<int>> offsets(solver->num_partition);
        for (auto &&partition_offsets : offsets)
            partition_offsets.resize(solver->num_partition, start);
        std::vector<std::vector<std::pair<int, Index>>> head_chains(solver->random_walk_batch_size);
        std::vector<std::vector<std::pair<int, Index>>> tail_chains(solver->random_walk_batch_size);
        for (auto &&head_chain : head_chains)
            head_chain.resize(solver->random_walk_length + 1);
        for (auto &&tail_chain : tail_chains)
            tail_chain.resize(solver->random_walk_length + 1);
        std::vector<int> sample_lengths(solver->random_walk_batch_size);
        int num_complete = 0, rand_id = 0;
        while (num_complete < solver->num_partition * solver->num_partition) {
            for (int i = 0; i < solver->random_walk_batch_size; i++) {
                if (rand_id > kRandBatchSize - solver->random_walk_length * 2) {
                    random.to_host();
                    CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));
                    rand_id = 0;
                }
                size_t edge_id = solver->edge_table.sample(random[rand_id++], random[rand_id++]);

                Index current = std::get<0>(solver->graph->edges[edge_id]);
                head_chains[i][0] = solver->head_locations[current];
                tail_chains[i][0] = solver->tail_locations[current];
                current = std::get<1>(solver->graph->edges[edge_id]);
                head_chains[i][1] = solver->head_locations[current];
                tail_chains[i][1] = solver->tail_locations[current];
                sample_lengths[i] = solver->random_walk_length;
                for (int j = 2; j <= solver->random_walk_length; j++)
                    if (!solver->graph->vertex_edges[current].empty()) {
                        Index neighbor_id = solver->vertex_edge_tables[current].sample(
                                random[rand_id++], random[rand_id++]);
                        current = std::get<0>(solver->graph->vertex_edges[current][neighbor_id]);
                        head_chains[i][j] = solver->head_locations[current];
                        tail_chains[i][j] = solver->tail_locations[current];
                    } else {
                        sample_lengths[i] = j - 1;
                        break;
                    }
            }
            for (int i = 0; i < solver->random_walk_batch_size; i++) {
                for (int j = 0; j < sample_lengths[i]; j++) {
                    for (int k = 1; k <= solver->augmentation_step; k++) {
                        if (j + k > sample_lengths[i])
                            break;
                        int head_partition_id = head_chains[i][j].first;
                        int tail_partition_id = tail_chains[i][j + k].first;
                        int &offset = offsets[head_partition_id][tail_partition_id];
                        if (offset < end) {
                            auto &pool = sample_pool[head_partition_id][tail_partition_id];
                            Index head_local_id = head_chains[i][j].second;
                            Index tail_local_id = tail_chains[i][j + k].second;
                            // pseudo shuffle
                            int shuffled_offset = offset % solver->shuffle_base * (pool_size / solver->shuffle_base)
                                                  + offset / solver->shuffle_base;
                            pool[shuffled_offset] = std::make_tuple(head_local_id, tail_local_id);
                            if (++offset == end)
                                num_complete++;
                        }
                    }
                }
            }
        }
    }
};

/** Training worker for graphs */
template<class _Solver>
class GraphWorker : public WorkerMixin<_Solver> {
public:
    typedef WorkerMixin<_Solver> Base;
    USING_WORKER_MIXIN(Base);
    using Base::Base;

    typedef GraphSolver<Solver::dim, Float, Index> GraphSolver;

    /**
     * Call the corresponding GPU kernel
     * (DeepWalk, LINE, node2vec) * (SGD, Momentum, AdaGrad, RMSprop, Adam)
     */
    bool kernel_dispatch() override {
        using namespace gpu;
        GraphSolver *solver = reinterpret_cast<GraphSolver *>(this->solver);

        switch (num_moment) {
            case 0: {
                decltype(&line::train<Vector, Index, kSGD>) train = nullptr;
                if (solver->model == "DeepWalk") {
                    if (optimizer.type == "SGD")
                        train = &deepwalk::train<Vector, Index, kSGD>;
                }
                if (solver->model == "LINE") {
                    if (optimizer.type == "SGD")
                        train = &line::train<Vector, Index, kSGD>;
                }
                if (solver->model == "node2vec") {
                    if (optimizer.type == "SGD")
                        train = &node2vec::train<Vector, Index, kSGD>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1],
                                    batch, negative_batch, optimizer, solver->negative_weight
#ifdef USE_LOSS
                            , this->loss
#endif
                    );
                    return true;
                }
            }
            case 1: {
                decltype(&line::train_1_moment<Vector, Index, kMomentum>) train = nullptr;
                if (solver->model == "DeepWalk") {
                    if (optimizer.type == "Momentum")
                        train = &deepwalk::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &deepwalk::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &deepwalk::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (solver->model == "LINE") {
                    if (optimizer.type == "Momentum")
                        train = &line::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &line::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &line::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (solver->model == "node2vec") {
                    if (optimizer.type == "Momentum")
                        train = &node2vec::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &node2vec::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &node2vec::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1],
                                    (*moments[0])[0], (*moments[1])[0],
                                    batch, negative_batch, optimizer, solver->negative_weight
#ifdef USE_LOSS
                            , this->loss
#endif
                    );
                    return true;
                }
            }
            case 2: {
                decltype(&line::train_2_moment<Vector, Index, kAdam>) train = nullptr;
                if (solver->model == "DeepWalk") {
                    if (optimizer.type == "Adam")
                        train = &deepwalk::train_2_moment<Vector, Index, kAdam>;
                }
                if (solver->model == "LINE") {
                    if (optimizer.type == "Adam")
                        train = &line::train_2_moment<Vector, Index, kAdam>;
                }
                if (solver->model == "node2vec") {
                    if (optimizer.type == "Adam")
                        train = &node2vec::train_2_moment<Vector, Index, kAdam>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1],
                                    (*moments[0])[0], (*moments[1])[0], (*moments[0])[1], (*moments[1])[1],
                                    batch, negative_batch, optimizer, solver->negative_weight
#ifdef USE_LOSS
                            , this->loss
#endif
                    );
                    return true;
                }
            }
        }
        return false;
    }
};

/**
 * @brief Node embedding solver
 * @tparam _dim dimension of embeddings
 * @tparam _Float floating type of parameters
 * @tparam _Index integral type of node indexes
 */
template<size_t _dim, class _Float = float, class _Index = size_t>
class GraphSolver :
        public SolverMixin<_dim, _Float, _Index, Graph, GraphSampler, GraphWorker> {
public:
    typedef SolverMixin<_dim, _Float, _Index, Graph, GraphSampler, GraphWorker> Base;
    USING_SOLVER_MIXIN(Base);
    using Base::Base;

    int augmentation_step, random_walk_length, random_walk_batch_size, shuffle_base;
    float p, q;
    std::shared_ptr<std::vector<Vector>> vertex_embeddings, context_embeddings;

    std::vector<AliasTable<Float, Index>> vertex_edge_tables;
    std::vector<AliasTable<Float, Index>> edge_edge_tables;

    GraphSolver(std::vector<int> device_ids = {}, int num_sampler_per_worker = kAuto, size_t gpu_memory_limit = kAuto):
            Base(device_ids, num_sampler_per_worker, gpu_memory_limit) {}

    /**
     * @brief Return the protocols of embeddings
     *
     * Vertex / context embeddings are binded to head / tail partitions respectively.
     * Both embeddings are updated in place.
     */
    inline std::vector<Protocol> get_protocols() const override {
        return {kHeadPartition | kInPlace, kTailPartition | kInPlace};
    };

    /** Return the protocol of negative sampling */
    inline Protocol get_sampler_protocol() const override {
        return kTailPartition;
    }

    /**
     * @brief Return the shapes of embeddings
     *
     * Shapes of both vertex and context embeddings can be inferred from the graph.
     */
    inline std::vector<Index> get_shapes() const override {
        return {kAuto, kAuto};
    }

    /** Return all available models of the solver */
    inline std::set<std::string> get_available_models() const override {
        return {"DeepWalk", "LINE", "node2vec"};
    }

    /** Return the default optimizer type and its hyperparameters */
    inline Optimizer get_default_optimizer() const override {
        return SGD(0.025, 5e-3);
    }

    /** Build alias reference for embeddings */
    inline void build_alias() override {
        vertex_embeddings = embeddings[0];
        context_embeddings = embeddings[1];
    }

    /** Build vertex edge tables. This function can be parallelized. */
    void build_vertex_edge(Index start, Index end) {
        for (Index i = start; i < end; i++) {
            std::vector<Float> vertex_edge_weights;
            for (auto &&edge : graph->vertex_edges[i])
                vertex_edge_weights.push_back(std::get<1>(edge));
            if (!vertex_edge_weights.empty())
                vertex_edge_tables[i].build(vertex_edge_weights);
        }
    }

    /** Build edge edge tables. This function can be parallelized. */
    void build_edge_edge(size_t start, size_t end, const std::vector<std::unordered_set<Index>> &neighbors) {
        for (size_t i = start; i < end; i++) {
            Index u = std::get<0>(graph->edges[i]);
            Index v = std::get<1>(graph->edges[i]);
            std::vector<Float> edge_edge_weights;
            for (auto &&edge : graph->vertex_edges[v]) {
                Index x = std::get<0>(edge);
                Float w = std::get<1>(edge);
                if (x == u)
                    edge_edge_weights.push_back(w / p);
                else if (neighbors[x].find(u) == neighbors[x].end())
                    edge_edge_weights.push_back(w / q);
                else
                    edge_edge_weights.push_back(w);
            }
            if (!edge_edge_weights.empty())
                edge_edge_tables[i].build(edge_edge_weights);
            CHECK(edge_edge_tables[i].count == graph->vertex_edges[v].size())
                    << "alias table count = " << edge_edge_tables[i].count
                    << ", vertex size = " << graph->vertex_edges[v].size();
        }
    }

    /** Determine and prepare the sampling function */
    SampleFunction get_sample_function() override {
        if (augmentation_step == 1)
            return Base::get_sample_function();

        graph->flatten();
        edge_table.build(graph->edge_weights);

        std::vector<std::thread> build_threads(num_thread);
        if (model == "DeepWalk" || model == "LINE") {
            for (Index i = 0; i < num_vertex; i++)
                vertex_edge_tables.push_back(AliasTable<Float, Index>(-1));
            Index work_load = (num_vertex + num_thread - 1) / num_thread;
            for (int i = 0; i < num_thread; i++)
                build_threads[i] = std::thread(&GraphSolver::build_vertex_edge, this, work_load * i,
                                               std::min(work_load * (i + 1), num_vertex));
            for (auto &&thread : build_threads)
                thread.join();
            return &Base::Sampler::sample_random_walk;
        }
        if (model == "node2vec") {
            std::vector<std::unordered_set<Index>> neighbors;
            for (auto &&vertex_edge : graph->vertex_edges) {
                std::unordered_set<Index> neighbor;
                for (auto &&edge : vertex_edge)
                    neighbor.insert(std::get<0>(edge));
                neighbors.push_back(neighbor);
            }

            size_t num_directed_edge = graph->edges.size();
            for (size_t i = 0; i < num_directed_edge; i++)
                edge_edge_tables.push_back(AliasTable<Float, Index>(-1));
            size_t work_load = (num_directed_edge + num_thread - 1) / num_thread;
            for (int i = 0; i < num_thread; i++)
                build_threads[i] = std::thread(&GraphSolver::build_edge_edge, this, work_load * i,
                                               std::min(work_load * (i + 1), num_directed_edge), neighbors);
            for (auto &&thread : build_threads)
                thread.join();
            return &Base::Sampler::sample_biased_random_walk;
        }

        return nullptr;
    }

    /** Initialize the embeddings */
    void init_embeddings() override {
        std::uniform_real_distribution<Float> init(-0.5 / dim, 0.5 / dim);
        for (auto &&embedding : *vertex_embeddings)
            for (int i = 0; i < dim; i++)
                embedding[i] = init(seed);
        for (auto &&embedding : *context_embeddings)
            embedding = 0;
    }

    inline std::string name() const override {
        std::stringstream ss;
        ss << "GraphSolver<" << dim << ", "
           << pretty::type2name<Float>() << ", " << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    inline std::string sampling_info() const override {
        std::stringstream ss;
        if (model == "LINE")
            ss << "augmentation step: " << augmentation_step << ", shuffle base: " << shuffle_base << std::endl;
        if (model == "DeepWalk")
            ss << "augmentation step: " << augmentation_step << std::endl;
        if (model == "node2vec")
            ss << "augmentation step: " << augmentation_step << ", p: " << p << ", q: " << q << std::endl;
        ss << "random walk length: " << random_walk_length << std::endl;
        ss << "random walk batch size: " << random_walk_batch_size << std::endl;
        ss << "#negative: " << num_negative << ", negative sample exponent: " << negative_sample_exponent;
        return ss.str();
    }

    /**
     * @brief Train node embeddings
     * @param _model "DeepWalk", "LINE" or "node2vec"
     * @param _num_epoch number of epochs, i.e. #positive edges / |E|
     * @param _resume resume training from learned embeddings or not
     * @param _augmentation_step node pairs with distance <= augmentation_step are considered as positive samples
     * @param _random_walk_length length of each random walk
     * @param _random_walk_batch_size batch size of random walks in samplers
     * @param _shuffle_base base for pseudo shuffle
     * @param _p return parameter (for node2vec)
     * @param _q in-out parameter (for node2vec)
     * @param _positive_reuse times of reusing positive samples
     * @param _negative_sample_exponent exponent of degrees in negative sampling
     * @param _negative_weight weight for each negative sample
     * @param _log_frequency log every log_frequency batches
     */
    void train(const std::string &_model = "LINE", int _num_epoch = 2000, bool _resume = false,
               int _augmentation_step = 5, int _random_walk_length = 40, int _random_walk_batch_size = 100,
               int _shuffle_base = kAuto, float _p = 1, float _q = 1, int _positive_reuse = 1,
               float _negative_sample_exponent = 0.75, float _negative_weight = 5, int _log_frequency = 1000) {
        augmentation_step = _augmentation_step;
        random_walk_length = _random_walk_length;
        random_walk_batch_size = _random_walk_batch_size;
        shuffle_base = _shuffle_base;
        p = _p;
        q = _q;

        if (shuffle_base == kAuto)
            shuffle_base = augmentation_step;
        if (model == "DeepWalk" || model == "node2vec")
            shuffle_base = 1;
        CHECK(augmentation_step >= 1) << "`augmentation_step` should be a positive integer";
        CHECK(augmentation_step <= random_walk_length)
                << "`random_walk_length` should be no less than `augmentation_step`";

        Base::train(_model, _num_epoch, _resume, random_walk_length * random_walk_batch_size, _positive_reuse,
                    _negative_sample_exponent, _negative_weight, _log_frequency);
    }

    /** Save vertex embeddings in word2vec format */
    void save_embeddings(const char *file_name) const {
        FILE *fout = fopen(file_name, "w");
        fprintf(fout, "%llu %llu\n", static_cast<unsigned long long>(num_vertex), static_cast<unsigned long long>(dim));
        for (Index i = 0; i < num_vertex; i++) {
            fprintf(fout, "%s ", graph->id2name[i].c_str());
            fwrite((*vertex_embeddings)[i].data, sizeof(Float), dim, fout);
            fprintf(fout, "\n");
        }
        fclose(fout);
    }
};

} // namespace graphvite