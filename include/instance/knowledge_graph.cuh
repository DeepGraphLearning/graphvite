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

#include <unordered_map>

#include "core/graph.h"
#include "core/solver.h"
#include "gpu/knowledge_graph.cuh"

/**
 * @page Knowledge Graph Embedding
 *
 * Knowledge graph embedding is an instantiation of the system on knowledge graphs (e.g. freebase, wordnet)
 *
 * In knowledge graph embedding, there are two embedding matrices,
 * namely the entity embeddings and the relation embeddings.
 * Since entity embeddings are related to both head and tail partitions, we implement them as two embedding protocols,
 * with the same underlying storage.
 * The workers generate negative entities for both heads and tails on GPUs.
 * To improve the performance of embedding, we adopt self-adversarial negative sampling to assign more weight for
 * hard negative samples.
 *
 * Currently, our knowledge graph embedding solver supports TransE, DistMult, ComplEx, SimplE and RotatE.
 *
 * Reference:
 *
 * 1) TransE
 * http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
 *
 * 2) DistMult
 * https://arxiv.org/pdf/1412.6575.pdf
 *
 * 3) ComplEx
 * http://proceedings.mlr.press/v48/trouillon16.pdf
 *
 * 4) SimplE
 * https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
 *
 * 5) RotatE
 * https://arxiv.org/pdf/1902.10197.pdf
 */

namespace graphvite {

/**
 * @brief Knowledge graphs
 * @tparam _Index integral type of node indexes
 */
template<class _Index = size_t>
class KnowledgeGraph : public GraphMixin<_Index, _Index> {
public:
    typedef GraphMixin<_Index, _Index> Base;
    USING_GRAPH_MIXIN(Base);

    typedef _Index Index;

    std::unordered_map<std::string, Index> entity2id, relation2id;
    std::vector<std::string> id2entity, id2relation;

    Index num_relation;
    bool normalization;

    /** Clear the graph and free CPU memory */
    void clear() override {
        Base::clear();
        decltype(entity2id)().swap(entity2id);
        decltype(relation2id)().swap(relation2id);
        decltype(id2entity)().swap(id2entity);
        decltype(id2relation)().swap(id2relation);
    }

    /** Normalize the adjacency matrix symetrically **/
    void normalize() {
        std::vector<std::unordered_map<Index, float>> head_weights(num_vertex), tail_weights(num_vertex);
        for (Index h = 0; h < num_vertex; h++)
            for (auto &&vertex_edge : vertex_edges[h]) {
                Index t = std::get<0>(vertex_edge);
                float w = std::get<1>(vertex_edge);
                Index r = std::get<2>(vertex_edge);
                if (head_weights[h].find(r) == head_weights[h].end())
                    head_weights[h][r] = 0;
                if (tail_weights[t].find(r) == tail_weights[t].end())
                    tail_weights[t][r] = 0;
                head_weights[h][r] += w;
                tail_weights[t][r] += w;
            }
        for (Index h = 0; h < num_vertex; h++) {
            float weight = 0;
            for (auto &&vertex_edge : vertex_edges[h]) {
                Index t = std::get<0>(vertex_edge);
                float &w = std::get<1>(vertex_edge);
                Index r = std::get<2>(vertex_edge);
                w /= sqrt(head_weights[h][r] * tail_weights[t][r]);
                weight += w;
            }
            vertex_weights[h] = weight;
        }
    }

    inline std::string name() const override {
        std::stringstream ss;
        ss << "KnowledgeGraph<" << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    inline std::string graph_info() const override {
        std::stringstream ss;
        ss << "#entity: " << num_vertex << ", #relation: " << num_relation << std::endl;
        ss << "#triplet: " << num_edge << ", normalization: " << pretty::yes_no(normalization);
        return ss.str();
    }

    /** Add an edge to the adjacency list */
    void add_edge(const std::string &h_name, const std::string &r_name, const std::string &t_name, float w) {
        Index h, t, r;
        auto h_iter = entity2id.find(h_name);
        if (h_iter != entity2id.end())
            h = h_iter->second;
        else {
            h = num_vertex++;
            entity2id[h_name] = h;
            id2entity.push_back(h_name);
            vertex_edges.push_back(std::vector<VertexEdge>());
            vertex_weights.push_back(0);
        }
        auto r_iter = relation2id.find(r_name);
        if (r_iter != relation2id.end())
            r = r_iter->second;
        else {
            r = num_relation++;
            relation2id[r_name] = r;
            id2relation.push_back(r_name);
        }
        auto t_iter = entity2id.find(t_name);
        if (t_iter != entity2id.end())
            t = t_iter->second;
        else {
            t = num_vertex++;
            entity2id[t_name] = t;
            id2entity.push_back(t_name);
            vertex_edges.push_back(std::vector<VertexEdge>());
            vertex_weights.push_back(0);
        }
        vertex_edges[h].push_back(std::make_tuple(t, w, r));
        vertex_weights[h] += w;
        num_edge++;
    }

    /**
     * @brief Load a knowledge graph from a triplet-list file. Store the graph in an adjacency list.
     * @param file_name file name
     * @param _normalization normalize the adjacency matrix or not
     * @param delimiters string of delimiter characters
     * @param comment prefix of comment strings
     */
    void load_file(const char *file_name, bool _normalization = false, const char *delimiters = " \t\r\n",
                   const char *comment = "#") {
        LOG(INFO) << "loading knowledge graph from " << file_name;
        clear();
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

            char *h_name = strtok(line, delimiters);
            if (!h_name)
                continue;
            char *r_name = strtok(nullptr, delimiters);
            char *t_name = strtok(nullptr, delimiters);
            char *w_str = strtok(nullptr, delimiters);
            char *more = strtok(nullptr, delimiters);
            CHECK(t_name && !more) << "Invalid format at line " << i;

            float w = w_str ? atof(w_str) : 1;
            add_edge(h_name, r_name, t_name, w);
        }
        fclose(fin);
        if (normalization)
            normalize();

        LOG(WARNING) << pretty::block(info());
    }

    /**
     * @brief Load a knowledge graph from a triplet list. Store the graph in an adjacency list.
     * @param triplet_list triplet list
     * @param _normalization normalize the adjacency matrix or not
     */
    void load(const std::vector<std::tuple<std::string, std::string, std::string, float>> &triplet_list,
              bool _normalization = false) {
        clear();
        normalization = _normalization;

        for (auto &&edge : triplet_list) {
            auto &h_name = std::get<0>(edge);
            auto &r_name = std::get<1>(edge);
            auto &t_name = std::get<2>(edge);
            float w = std::get<3>(edge);

            add_edge(h_name, r_name, t_name, w);
        }
        if (normalization)
            normalize();

        LOG(WARNING) << pretty::block(info());
    }

    /**
     * @brief Save the graph in triplet-list format
     * @param file_name file name
     * @param anonymous save entity / relation names or not
     */
    void save(const char *file_name, bool anonymous = false) {
        LOG(INFO) << "Saving weighted graph to " << file_name;

        FILE *fout = fopen(file_name, "w");

        for (unsigned long long i = 0; i < num_vertex; i++)
            for (auto &&vertex_edge : vertex_edges[i]) {
                unsigned long long j = std::get<0>(vertex_edge);
                unsigned long long r = std::get<1>(vertex_edge);
                if (anonymous)
                    fprintf(fout, "%llu\t%llu\t%llu\n", i, j, r);
                else
                    fprintf(fout, "%s\t%s\t%s\n", id2entity[i].c_str(), id2entity[j].c_str(), id2relation[r].c_str());
            }
        fclose(fout);
    }
};

template <size_t _dim, class _Float, class _Index>
class KnowledgeGraphSolver;

/** Edge sampler for knowledge graphs */
template<class _Solver>
#define _Index typename _Solver::Index
class KnowledgeGraphSampler : public SamplerMixin<_Solver, _Index> {
public:
    typedef SamplerMixin<_Solver, _Index> Base;
#undef _Index
    USING_SAMPLER_MIXIN(Base);
    using Base::Base;

    /** Return the relation as additional attributes */
    inline Attributes get_attributes(const Edge &edge) const override {
        return Attributes(std::get<3>(edge));
    }
};

/** Training worker for knowledge graphs */
template<class _Solver>
class KnowledgeGraphWorker : public WorkerMixin<_Solver> {
public:
    typedef WorkerMixin<_Solver> Base;
    USING_WORKER_MIXIN(Base);
    using Base::Base;

    typedef KnowledgeGraphSolver<Solver::dim, Float, Index> KnowledgeGraphSolver;

    /** Build the alias table for negative sampling. Knowledge graphs use uniform negative sampling. */
    void build_negative_sampler() override {
        std::vector<Float> negative_weights(head_partition_size + tail_partition_size, 1);
        negative_sampler.build(negative_weights);
    }

    /**
     * Call the corresponding GPU kernel
     * (TransE, DistMult, ComplEx, SimplE, RotatE) * (SGD, Momentum, AdaGrad, RMSprop, Adam)
     */
    bool kernel_dispatch() override {
        using namespace gpu;
        KnowledgeGraphSolver *solver = reinterpret_cast<KnowledgeGraphSolver *>(this->solver);

        float margin_or_l3;
        if (solver->model == "TransE" || solver->model == "RotatE")
            margin_or_l3 = solver->margin;
        if (solver->model == "DistMult" || solver->model == "ComplEx" || solver->model == "SimplE")
            margin_or_l3 = solver->l3_regularization;
        switch (num_moment) {
            case 0: {
                decltype(&transe::train<Vector, Index, kSGD>) train = nullptr;
                if (solver->model == "TransE") {
                    if (optimizer.type == "SGD")
                        train = &transe::train<Vector, Index, kSGD>;
                }
                if (solver->model == "DistMult") {
                    if (optimizer.type == "SGD")
                        train = &distmult::train<Vector, Index, kSGD>;
                }
                if (solver->model == "ComplEx") {
                    if (optimizer.type == "SGD")
                        train = &complex::train<Vector, Index, kSGD>;
                }
                if (solver->model == "SimplE") {
                    if (optimizer.type == "SGD")
                        train = &simple::train<Vector, Index, kSGD>;
                }
                if (solver->model == "RotatE") {
                    if (optimizer.type == "SGD")
                        train = &rotate::train<Vector, Index, kSGD>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1], *embeddings[2], *gradients[2],
                                    batch, negative_batch, optimizer, margin_or_l3, solver->adversarial_temperature
#ifdef USE_LOSS
                            , this->loss
#endif
                    );
                    return true;
                }
                break;
            }
            case 1: {
                decltype(&transe::train_1_moment<Vector, Index, kMomentum>) train = nullptr;
                if (solver->model == "TransE") {
                    if (optimizer.type == "Momentum")
                        train = &transe::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &transe::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &transe::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (solver->model == "DistMult") {
                    if (optimizer.type == "Momentum")
                        train = &distmult::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &distmult::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &distmult::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (solver->model == "ComplEx") {
                    if (optimizer.type == "Momentum")
                        train = &complex::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &complex::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &complex::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (solver->model == "SimplE") {
                    if (optimizer.type == "Momentum")
                        train = &simple::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &simple::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &simple::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (solver->model == "RotatE") {
                    if (optimizer.type == "Momentum")
                        train = &rotate::train_1_moment<Vector, Index, kMomentum>;
                    if (optimizer.type == "AdaGrad")
                        train = &rotate::train_1_moment<Vector, Index, kAdaGrad>;
                    if (optimizer.type == "RMSprop")
                        train = &rotate::train_1_moment<Vector, Index, kRMSprop>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1], *embeddings[2], *gradients[2],
                                    (*moments[0])[0], (*moments[1])[0], (*moments[2])[0],
                                    batch, negative_batch, optimizer, margin_or_l3, solver->adversarial_temperature
#ifdef USE_LOSS
                            , this->loss
#endif
                    );
                    return true;
                }
                break;
            }
            case 2: {
                decltype(&transe::train_2_moment<Vector, Index, kAdam>) train = nullptr;
                if (solver->model == "TransE") {
                    if (optimizer.type == "Adam")
                        train = &transe::train_2_moment<Vector, Index, kAdam>;
                }
                if (solver->model == "DistMult") {
                    if (optimizer.type == "Adam")
                        train = &distmult::train_2_moment<Vector, Index, kAdam>;
                }
                if (solver->model == "ComplEx") {
                    if (optimizer.type == "Adam")
                        train = &complex::train_2_moment<Vector, Index, kAdam>;
                }
                if (solver->model == "SimplE") {
                    if (optimizer.type == "Adam")
                        train = &simple::train_2_moment<Vector, Index, kAdam>;
                }
                if (solver->model == "RotatE") {
                    if (optimizer.type == "Adam")
                        train = &rotate::train_2_moment<Vector, Index, kAdam>;
                }
                if (train) {
                    train<<<kBlockPerGrid, kThreadPerBlock, 0, work_stream>>>
                            (*embeddings[0], *embeddings[1], *embeddings[2], *gradients[2],
                                    (*moments[0])[0], (*moments[1])[0], (*moments[2])[0],
                                    (*moments[0])[1], (*moments[1])[1], (*moments[2])[1],
                                    batch, negative_batch, optimizer, margin_or_l3, solver->adversarial_temperature
#ifdef USE_LOSS
                            , this->loss
#endif
                    );
                    return true;
                }
                break;
            }
        }
        return false;
    }
};

/**
 * @brief Knowledge graph embedding solver
 * @tparam _dim dimension of embeddings
 * @tparam _Float floating type of parameters
 * @tparam _Index integral type of node indexes
 */
template <size_t _dim, class _Float = float, class _Index = size_t>
class KnowledgeGraphSolver :
        public SolverMixin<_dim, _Float, _Index, KnowledgeGraph, KnowledgeGraphSampler, KnowledgeGraphWorker> {
public:
    typedef SolverMixin<_dim, _Float, _Index, KnowledgeGraph, KnowledgeGraphSampler, KnowledgeGraphWorker> Base;
    USING_SOLVER_MIXIN(Base);
    using Base::Base;

    float margin, l3_regularization;
    float adversarial_temperature;
    std::shared_ptr<std::vector<Vector>> entity_embeddings, relation_embeddings;

    /**
     * @brief Return the protocols of embeddings
     *
     * Head / tail entity embeddings are binded to head / tail partitions respectively.
     * The two embeddings share the same underlying storage. They are updated in place.
     * The relation embeddings are binded to global range.
     * It is updated by summed gradients at the end of each epsiode.
     */
    inline std::vector<Protocol> get_protocols() const override {
        return {kHeadPartition | kInPlace, kTailPartition | kInPlace | kSharedWithPredecessor, kGlobal};
    }

    /** Return the protocol of negative sampling */
    inline Protocol get_sampler_protocol() const override {
        return kHeadPartition | kTailPartition;
    }

    /**
     * @brief Return the shapes of embeddings
     *
     * Shapes of both head and tail entity embeddings can be inferred from the graph.
     * The shape of relation embeddings equals to the number of relations in graph.
     */
    inline std::vector<Index> get_shapes() const override {
        return {kAuto, kAuto, graph->num_relation};
    }

    /** Return all available models of the solver */
    inline std::set<std::string> get_available_models() const override {
        return {"TransE", "RotatE", "DistMult", "ComplEx", "SimplE"};
    }

    /** Return the default optimizer type and its hyperparameters */
    inline Optimizer get_default_optimizer() const override {
        return Adam(1e-4, 0);
    }

    /** Build alias reference for embeddings */
    inline void build_alias() override {
        entity_embeddings = embeddings[0];
        relation_embeddings = embeddings[2];
    }

    /** Initialize the embeddings */
    void init_embeddings() override {
        static const Float kPi = atan(1) * 4;

        if (model == "TransE") {
            std::uniform_real_distribution<Float> init(-margin / dim, margin / dim);
            for (auto &&embedding : *entity_embeddings)
                for (int i = 0; i < dim; i++)
                    embedding[i] = init(seed);
            for (auto &&embedding : *relation_embeddings)
                for (int i = 0; i < dim; i++)
                    embedding[i] = init(seed);
        }
        if (model == "DistMult" || model == "ComplEx" || model == "SimplE") {
            std::uniform_real_distribution<Float> init(-0.5, 0.5);
            for (auto &&embedding : *entity_embeddings)
                for (int i = 0; i < dim; i++)
                    embedding[i] = init(seed);
            for (auto &&embedding : *relation_embeddings)
                for (int i = 0; i < dim; i++)
                    embedding[i] = init(seed);
        }
        if (model == "RotatE") {
            std::uniform_real_distribution<Float> init(-margin * 2 / dim, margin * 2 / dim);
            std::uniform_real_distribution<Float> init_phase(-kPi, kPi);
            for (auto &&embedding : *entity_embeddings)
                for (int i = 0; i < dim; i++)
                    embedding[i] = init(seed);
            for (auto &&embedding : *relation_embeddings)
                for (int i = 0; i < dim / 2; i++)
                    embedding[i] = init_phase(seed);
        }
    }

    inline std::string name() const override {
        std::stringstream ss;
        ss << "KnowledgeGraphSolver<" << dim << ", "
           << pretty::type2name<Float>() << ", " << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    inline std::string sampling_info() const override {
        std::stringstream ss;
        ss << "positive sample batch size: " << sample_batch_size << std::endl;
        ss << "#negative: " << num_negative;
        return ss.str();
    }

    inline std::string training_info() const override {
        std::stringstream ss;
        ss << "model: " << model << std::endl;
        ss << optimizer.info() << std::endl;
        ss << "#epoch: " << num_epoch << ", batch size: " << batch_size << std::endl;
        ss << "resume: " << pretty::yes_no(resume) << std::endl;
        if (model == "TransE" || model == "RotatE")
            ss << "margin: " << margin << ", positive reuse: " << positive_reuse << std::endl;
        if (model == "DistMult" || model == "ComplEx" || model == "SimplE")
            ss << "l3 regularization: " << l3_regularization << ", positive reuse: " << positive_reuse << std::endl;
        ss << "adversarial temperature: " << adversarial_temperature;
        return ss.str();
    }

    /**
     * @brief Train knowledge graph embeddings
     * @param _model "TransE", "DistMult", "ComplEx", "SimplE" or "RotatE"
     * @param _num_epoch number of epochs, i.e. #positive edges / |E|
     * @param _resume resume training from learned embeddings or not
     * @param _margin logit margin (for TransE & RotatE)
     * @param _l3_regularization l3 regularization (for DistMult, ComplEx & SimplE)
     * @param _sample_batch_size batch size of samples in samplers
     * @param _positive_reuse times of reusing positive samples
     * @param _adversarial_temperature temperature of adversarial negative sampling,
     *     disabled when set to non-positive value
     * @param _log_frequency log every log_frequency batches
     */
    void train(const std::string &_model = "RotatE", int _num_epoch = 2000, bool _resume = false, float _margin = 24,
               float _l3_regularization = 2e-3, int _sample_batch_size = 2000, int _positive_reuse = 1,
               float _adversarial_temperature = 2, int _log_frequency = 100) {
        margin = _margin;
        l3_regularization = _l3_regularization;
        adversarial_temperature = _adversarial_temperature;

        Base::train(_model, _num_epoch, _resume, _sample_batch_size, _positive_reuse, 0, 1.0f / num_negative,
                    _log_frequency);
    }
};

} // namespace graphvite