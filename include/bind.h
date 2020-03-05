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

#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "base/vector.h"
#include "instance/graph.cuh"
#include "instance/word_graph.cuh"
#include "instance/knowledge_graph.cuh"
#include "instance/visualization.cuh"

#include "util/io.h"

namespace py = pybind11;

namespace pybind11 {
typedef call_guard<gil_scoped_release> no_gil;

#if (PY_MAJOR_VERSION == 2)
    typedef bytes internal_str;
#else
    typedef str internal_str;
#endif
} // namespace pybind11

enum DType {
    uint32 = 0,
    uint64,
    float32,
    float64
};

namespace std {

template<>
struct hash<DType>
{
    size_t operator()(const DType& dtype) const noexcept
    { return static_cast<size_t>(dtype); }
};

}

const std::unordered_map<DType, std::string> dtype2name = {
        {uint32, typeid(unsigned int).name()},
        {uint64, typeid(unsigned long int).name()},
        {float32, typeid(float).name()},
        {float64, typeid(double).name()}
};

std::ostream& operator <<(std::ostream &out, DType dtype) {
    return out << dtype2name.find(dtype)->second;
}

template<class... Args>
std::string signature(const std::string &name, const Args &...args) {
    std::stringstream full_name;
    full_name << name;
    auto _ = {(full_name << "_" << args, 0)...};
    return full_name.str();
}

template<class Solver>
#define Float typename Solver::Float
#define Vector typename Solver::Vector
std::function<py::array_t<Float>(Solver &)> numpy_view(std::shared_ptr<std::vector<Vector>> Solver::*field) {
#undef Vector
#undef Float
    const size_t dim = Solver::dim;
    typedef typename Solver::Float Float;

    return [field](Solver &solver) {
        Float *data = reinterpret_cast<Float *>((solver.*field)->data());
        py::capsule dummy(data, [](void *ptr) {});
        return py::array_t<Float>({(solver.*field)->size(), dim},
                                  {sizeof(Float) * dim, sizeof(Float)},
                                  data, dummy);
    };
}

// Class interface
template<class Index = size_t>
class pyGraph : public py::class_<graphvite::Graph<Index>> {
public:
    typedef graphvite::Graph<Index> Graph;
    typedef py::class_<Graph> Base;
    using Base::attr;
    using Base::def;
    using Base::def_readonly;

    template<class... Args>
    pyGraph(py::handle scope, const char *name, const Args &...args) :
            Base(scope, signature(name, typeid(Index).name()).c_str(), args...) {
        attr("__doc__") = "Graph(index_type=dtype.uint32)"
        R"(
        Normal graphs without attributes.

        Parameters:
            index_type (dtype): type of node indexes
        )";

        // override instance name with template name
        attr("__name__") = py::internal_str(name);
        attr("__qualname__") = py::internal_str(name);

        // data members
        def_readonly("num_vertex", &Graph::num_vertex);
        def_readonly("num_edge", &Graph::num_edge);
        def_readonly("as_undirected", &Graph::as_undirected);
        def_readonly("normalization", &Graph::normalization);
        def_readonly("name2id", &Graph::name2id, "Map of node name to index.");
        def_readonly("id2name", &Graph::id2name, "Map of node index to name.");

        // member functions
        def(py::init<>(), py::no_gil());

        def("load", &Graph::load_file, py::no_gil(),
            py::arg("file_name"), py::arg("as_undirected") = true, py::arg("normalization") = false,
            py::arg("delimiters") = " \t\r\n", py::arg("comment") = "#",
            "load(*args, **kwargs)"
            R"(
            Load a graph from an edge-list file. Store the graph in an adjacency list.

            This function has 3 overloads

            .. function:: load(file_name, as_undirected=True, normalization=False, delimiters=' \\t\\r\\n', comment='#')
            .. function:: load(edge_list, as_undirected=True, normalization=False)
            .. function:: load(weighted_edge_list, as_undirected=True, normalization=False)

            Parameters:
                file_name (str): file name
                edge_list (list of (str, str)): edge list
                weighted_edge_list (list of (str, str, float)): weighted edge list
                as_undirected (bool, optional): symmetrize the graph or not
                normalization (bool, optional): normalize the adjacency matrix or not
                delimiters (str, optional): string of delimiter characters
                comment (str, optional): prefix of comment strings
            )");

        def("load", &Graph::load_edge_list, py::no_gil(),
            py::arg("edge_list"), py::arg("as_undirected") = true, py::arg("normalization") = false);

        def("load", &Graph::load_weighted_edge_list, py::no_gil(),
            py::arg("weighted_edge_list"), py::arg("as_undirected") = true, py::arg("normalization") = false);

        def("save", &Graph::save, py::no_gil(),
            py::arg("file_name"), py::arg("weighted") = true, py::arg("anonymous") = false,
            "save(file_name, weighted=True, anonymous=False)"
            R"(
            Save the graph in edge-list format.

            Parameters:
                file_name (str): file name
                weighted (bool, optional): save edge weights or not
                anonymous (bool, optional): save node names or not
            )");

        def("__repr__", &Graph::info, py::no_gil());
    }
};

template<class Index = size_t>
class pyWordGraph : public py::class_<graphvite::WordGraph<Index>, graphvite::Graph<Index>> {
public:
    typedef graphvite::WordGraph<Index> WordGraph;
    typedef py::class_<WordGraph, graphvite::Graph<Index>> Base;
    using Base::attr;
    using Base::def;
    using Base::def_readonly;

    template<class... Args>
    pyWordGraph(py::handle scope, const char *name, const Args &...args) :
    Base(scope, signature(name, typeid(Index).name()).c_str(), args...) {
        attr("__doc__") = "WordGraph(index_type=dtype.uint32)"
        R"(
        Normal graphs of word co-occurrences.

        Parameters:
            index_type (dtype): type of node indexes
        )";

        // override instance name with template name
        attr("__name__") = py::internal_str(name);
        attr("__qualname__") = py::internal_str(name);

        // member functions
        def(py::init<>(), py::no_gil());

        def("load", &WordGraph::load_file_compact, py::no_gil(),
            py::arg("file_name"), py::arg("window") = 5, py::arg("min_count") = 5, py::arg("normalization") = false,
            py::arg("delimiters") = " \t\r\n", py::arg("comment") = "#",
            R"(load(file_name, window=5, min_count=5, normalization=False, delimiters=' \\t\\r\\n', comment='#'))"
            R"(
            Load a word graph from a corpus file. Store the graph in an adjacency list.

            Parameters:
                file_name (str): file name
                window (int, optional): word pairs with distance <= window are counted as edges
                min_count (int, optional): words with occurrence <= min_count are discarded
                normalization (bool, optional): normalize the adjacency matrix or not
                delimiters (str, optional): string of delimiter characters
                comment (str, optional): prefix of comment strings
            )");

        def("__repr__", &WordGraph::info, py::no_gil());
    }
};

template<class Index = size_t>
class pyKnowledgeGraph : public py::class_<graphvite::KnowledgeGraph<Index>> {
public:
    typedef graphvite::KnowledgeGraph<Index> KnowledgeGraph;
    typedef py::class_<KnowledgeGraph> Base;
    using Base::attr;
    using Base::def;
    using Base::def_readonly;

    template<class... Args>
    pyKnowledgeGraph(py::handle scope, const char *name, const Args &...args) :
            Base(scope, signature(name, typeid(Index).name()).c_str(), args...) {
        attr("__doc__") = "KnowledgeGraph(index_type=dtype.uint32)"
        R"(
        Knowledge graphs.

        Parameters:
            index_type (dtype): type of node indexes
        )";

        // override instance name with template name
        attr("__name__") = py::internal_str(name);
        attr("__qualname__") = py::internal_str(name);

        // data members
        def_readonly("num_vertex", &KnowledgeGraph::num_vertex);
        def_readonly("num_edge", &KnowledgeGraph::num_edge);
        def_readonly("num_relation", &KnowledgeGraph::num_relation);
        def_readonly("normalization", &KnowledgeGraph::normalization);
        def_readonly("entity2id", &KnowledgeGraph::entity2id, "Map of entity name to index.");
        def_readonly("relation2id", &KnowledgeGraph::relation2id, "Map of relation name to index.");
        def_readonly("id2entity", &KnowledgeGraph::id2entity, "Map of entity index to name.");
        def_readonly("id2relation", &KnowledgeGraph::id2relation, "Map of relation index to name.");

        // member functions
        def(py::init<>(), py::no_gil());

        def("load", &KnowledgeGraph::load_file, py::no_gil(),
            py::arg("file_name"), py::arg("normalization") = false, py::arg("delimiters") = " \t\r\n",
            py::arg("comment") = "#",
            "load(*args, **kwargs)"
            R"(
            Load a knowledge graph from a triplet-list file. Store the graph in an adjacency list.

            This function has 3 overloads

            .. function:: load(file_name, normalization=False, delimiters=' \\t\\r\\n', comment='#')
            .. function:: load(triplet_list, normalization=False)
            .. function:: load(weighted_triplet_list, normalization=False)

            Parameters:
                file_name (str): file name
                triplet_list (list of (str, str, str)): triplet list
                weighted_triplet_list (list of (str, str, str, float)): weighted triplet list
                normalization (bool, optional): normalize the adjacency matrix or not
                delimiters (str, optional): string of delimiter characters
                comment (str, optional): prefix of comment strings
            )");

        def("load", &KnowledgeGraph::load_triplet_list, py::no_gil(),
            py::arg("triplet_list"),  py::arg("normalization") = false);

        def("load", &KnowledgeGraph::load_weighted_triplet_list, py::no_gil(),
            py::arg("weighted_triplet_list"),  py::arg("normalization") = false);

        def("save", &KnowledgeGraph::save, py::no_gil(),
            py::arg("file_name"), py::arg("anonymous") = false,
            "save(file_name, anonymous=False)"
            R"(
            Save the graph in triplet-list format.

            Parameters:
                file_name (str): file name
                anonymous (bool, optional): save entity / relation names or not
            )");

        def("__repr__", &KnowledgeGraph::info, py::no_gil());
    }
};

template<class Index = size_t>
class pyKNNGraph : public py::class_<graphvite::KNNGraph<Index>, graphvite::Graph<Index>> {
public:
    typedef graphvite::KNNGraph<Index> KNNGraph;
    typedef py::class_<KNNGraph, graphvite::Graph<Index>> Base;
    using Base::attr;
    using Base::def;
    using Base::def_readonly;

    template<class... Args>
    pyKNNGraph(py::handle scope, const char *name, const Args &...args) :
            Base(scope, signature(name, typeid(Index).name()).c_str(), args...) {
        attr("__doc__") = "KNNGraph(index_type=dtype.uint32, device_ids=[], num_thread_per_worker=auto)"
        R"(
        K-nearest neighbor graphs.

        Parameters:
            index_type (dtype, optional): type of node indexes
            device_ids (list of int, optional): GPU ids, [] for auto
            num_thread_per_worker (int, optional): number of CPU thread per GPU
        )";

        // override instance name with template name
        attr("__name__") = py::internal_str(name);
        attr("__qualname__") = py::internal_str(name);

        // data members
        def_readonly("num_neighbor", &KNNGraph::num_neighbor);
        def_readonly("perplexity", &KNNGraph::perplexity);
        def_readonly("vector_normalization", &KNNGraph::vector_normalization);
        def_readonly("num_worker", &KNNGraph::num_worker);
        def_readonly("num_thread", &KNNGraph::num_thread);

        // member functions
        def(py::init<std::vector<int>, int>(), py::no_gil(),
            py::arg("device_ids") = std::vector<int>(), py::arg("num_thread_per_worker") = graphvite::kAuto);

        def("load", &KNNGraph::load_file, py::no_gil(),
            py::arg("vector_file"), py::arg("num_neighbor") = 200, py::arg("perplexity") = 30,
            py::arg("vector_normalization") = true, py::arg("delimiters") = " \t\r\n", py::arg("comment") = "#",
            "load(*arg, **kwargs)"
            R"(
            Build a KNN graph from a vector list. Store the graph in an adjacency list.

            This function has 2 overloads

            .. function:: load(vector_file, num_neighbor=200, perplexity=30, vector_normalization=True, delimiters=' \\t\\r\\n', comment='#')
            .. function:: load(vectors, num_neighbor=200, perplexity=30, vector_normalization=True)

            Parameters:
                file_name (str): file name
                vectors (2D array_like): vector list
                num_neighbor (int, optional): number of neighbors for each node
                perplexity (int, optional): perplexity for the neighborhood of each node
                vector_normalization (bool, optional): normalize the input vectors or not
                delimiters (str, optional): string of delimiter characters
                comment (str, optional): prefix of comment strings
            )");

        def("load", &KNNGraph::load_numpy, py::no_gil(),
            py::arg("vectors"), py::arg("num_neighbor") = 200, py::arg("perplexity") = 30,
            py::arg("vector_normalization") = true);

        def("__repr__", &KNNGraph::info, py::no_gil());
    }
};

template<size_t dim, class Float = float, class Index = size_t>
class pyGraphSolver : public py::class_<graphvite::GraphSolver<dim, Float, Index>> {
public:
    typedef graphvite::GraphSolver<dim, Float, Index> GraphSolver;
    typedef py::class_<GraphSolver> Base;
    using Base::attr;
    using Base::def;
    using Base::def_readonly;
    using Base::def_property_readonly;

    template<class... Args>
    pyGraphSolver(py::handle scope, const char *name, const Args &...args) :
            Base(scope, signature(name, dim, typeid(Float).name(), typeid(Index).name()).c_str(), args...) {
        attr("__doc__") = "GraphSolver(dim, float_type=dtype.float32, index_type=dtype.uint32, "
                                      "device_ids=[], num_sampler_per_worker=auto, gpu_memory_limit=auto)"
        R"(
        Graph embedding solver.

        Parameters:
            dim (int): dimension of embeddings
            float_type (dtype): type of parameters
            index_type (dtype): type of node indexes
            device_ids (list of int, optional): GPU ids, [] for auto
            num_sampler_per_worker (int, optional): number of sampler thread per GPU
            gpu_memory_limit (int, optional): memory limit for each GPU in bytes
        )";

        // override instance name with template name
        attr("__name__") = py::internal_str(name);
        attr("__qualname__") = py::internal_str(name);

        // data members
        def_readonly("num_partition", &GraphSolver::num_partition);
        def_readonly("num_negative", &GraphSolver::num_negative);
        def_readonly("optimizer", &GraphSolver::optimizer);
        def_readonly("negative_sample_exponent", &GraphSolver::negative_sample_exponent);
        def_readonly("negative_weight", &GraphSolver::negative_weight);
        def_readonly("model", &GraphSolver::model);
        def_readonly("num_epoch", &GraphSolver::num_epoch);
        def_readonly("resume", &GraphSolver::resume);
        def_readonly("episode_size", &GraphSolver::episode_size);
        def_readonly("batch_size", &GraphSolver::batch_size);
        def_readonly("augmentation_step", &GraphSolver::augmentation_step);
        def_readonly("random_walk_length", &GraphSolver::random_walk_length);
        def_readonly("random_walk_batch_size", &GraphSolver::random_walk_batch_size);
        def_readonly("shuffle_base", &GraphSolver::shuffle_base);
        def_readonly("p", &GraphSolver::p);
        def_readonly("q", &GraphSolver::q);
        def_readonly("positive_reuse", &GraphSolver::positive_reuse);
        def_readonly("log_frequency", &GraphSolver::log_frequency);
        def_readonly("num_worker", &GraphSolver::num_worker);
        def_readonly("num_sampler", &GraphSolver::num_sampler);
        def_readonly("gpu_memory_limit", &GraphSolver::gpu_memory_limit);
        def_readonly("gpu_memory_cost", &GraphSolver::gpu_memory_cost);

        // numpy views (mutable but unassignable)
        def_property_readonly("vertex_embeddings", numpy_view(&GraphSolver::vertex_embeddings),
                              "Vertex node embeddings (2D numpy view).");
        def_property_readonly("context_embeddings", numpy_view(&GraphSolver::context_embeddings),
                              "Context node embeddings (2D numpy view).");

        // member functions
        def(py::init<std::vector<int>, int, size_t>(), py::no_gil(),
            py::arg("device_ids") = std::vector<int>(), py::arg("num_sampler_per_worker") = graphvite::kAuto,
            py::arg("gpu_memory_limit") = graphvite::kAuto);

        def("build", &GraphSolver::build, py::no_gil(),
            py::arg("graph"), py::arg("optimizer") = graphvite::Optimizer(graphvite::kAuto),
            py::arg("num_partition") = graphvite::kAuto, py::arg("num_negative") = 1, py::arg("batch_size") = 100000,
            py::arg("episode_size") = graphvite::kAuto,
            "build(graph, optimizer=auto, num_partition=auto, num_negative=1, batch_size=100000, episode_size=auto)"
            R"(
            Determine and allocate all resources for the solver.

            Parameters:
                graph (Graph): graph
                optimizer (Optimizer or float, optional): optimizer or learning rate
                num_partition (int, optional): number of partitions
                num_negative (int, optional): number of negative samples per positive sample
                batch_size (int, optional): batch size of samples in CPU-GPU transfer
                episode_size (int, optional): number of batches in a partition block
            )");

        def("train", &GraphSolver::train, py::no_gil(),
            py::arg("model") = "LINE", py::arg("num_epoch") = 2000, py::arg("resume") = false,
            py::arg("augmentation_step") = graphvite::kAuto, py::arg("random_walk_length") = 40,
            py::arg("random_walk_batch_size") = 100, py::arg("shuffle_base") = graphvite::kAuto, py::arg("p") = 1,
            py::arg("q") = 1, py::arg("positive_reuse") = 1, py::arg("negative_sample_exponent") = 0.75,
            py::arg("negative_weight") = 5, py::arg("log_frequency") = 1000,
            "train(model='LINE', num_epoch=2000, resume=False, augmentation_step=auto, random_walk_length=40, "
                  "random_walk_batch_size=100, shuffle_base=auto, p=1, q=1, positive_reuse=1, "
                  "negative_sample_exponent=0.75, negative_weight=5, log_frequency=1000)"
            R"(
            Train node embeddings.

            Parameters:
                model (str, optional): 'DeepWalk', 'LINE' or 'node2vec'
                num_epoch (int, optional): number of epochs, i.e. #positive edges / \|E\|
                resume (bool, optional): resume training from learned embeddings or not
                augmentation_step (int, optional):
                    node pairs with distance <= augmentation_step are considered as positive samples
                random_walk_length (int, optional): length of each random walk
                random_walk_batch_size (int, optional): batch size of random walks in samplers
                shuffle_base (int, optional): base for pseudo shuffle
                p (float, optional): return parameter (for node2vec)
                q (float, optional): in-out parameter (for node2vec)
                positive_reuse (int, optional): times of reusing positive samples
                negative_sample_exponent (float, optional): exponent of degrees in negative sampling
                negative_weight (float, optional): weight for each negative sample
                log_frequency (int, optional): log every log_frequency batches
            )");

        def("predict", &GraphSolver::predict_numpy, py::no_gil(),
            py::arg("samples"),
            "predict(samples)"
            R"(
            Predict logits for samples.

            Parameters:
                samples (ndarray): triplets with shape (?, 2), each triplet is ordered as (v, c)
            )");

        def("clear", &GraphSolver::clear, py::no_gil(),
            "clear()"
            R"(
            Free CPU and GPU memory, except the embeddings on CPU.
            )");

        def("__repr__", &GraphSolver::info, py::no_gil());
    }
};

template<size_t dim, class Float = float, class Index = size_t>
class pyKnowledgeGraphSolver : public py::class_<graphvite::KnowledgeGraphSolver<dim, Float, Index>> {
public:
    typedef graphvite::KnowledgeGraphSolver<dim, Float, Index> KnowledgeGraphSolver;
    typedef py::class_<KnowledgeGraphSolver> Base;
    using Base::attr;
    using Base::def;
    using Base::def_readonly;
    using Base::def_property_readonly;

    template<class... Args>
    pyKnowledgeGraphSolver(py::handle scope, const char *name, const Args &...args) :
            Base(scope, signature(name, dim, typeid(Float).name(), typeid(Index).name()).c_str(), args...) {
        attr("__doc__") = "KnowledgeGraphSolver(dim, float_type=dtype.float32, index_type=dtype.uint32, "
                                               "device_ids=[], num_sampler_per_worker=auto, gpu_memory_limit=auto)"
        R"(
        Knowledge graph embedding solver.

        Parameters:
            dim (int): dimension of embeddings
            float_type (dtype): type of parameters
            index_type (dtype): type of node indexes
            device_ids (list of int, optional): GPU ids, [] for auto
            num_sampler_per_worker (int, optional): number of sampler thread per GPU
            gpu_memory_limit (int, optional): memory limit for each GPU in bytes
        )";

        // override instance name with template name
        attr("__name__") = py::internal_str(name);
        attr("__qualname__") = py::internal_str(name);

        // data members
        def_readonly("num_partition", &KnowledgeGraphSolver::num_partition);
        def_readonly("num_negative", &KnowledgeGraphSolver::num_negative);
        def_readonly("sample_batch_size", &KnowledgeGraphSolver::sample_batch_size);
        def_readonly("optimizer", &KnowledgeGraphSolver::optimizer);
        def_readonly("negative_sample_exponent", &KnowledgeGraphSolver::negative_sample_exponent);
        def_readonly("model", &KnowledgeGraphSolver::model);
        def_readonly("num_epoch", &KnowledgeGraphSolver::num_epoch);
        def_readonly("resume", &KnowledgeGraphSolver::resume);
        def_readonly("relation_lr_multiplier", &KnowledgeGraphSolver::relation_lr_multiplier);
        def_readonly("episode_size", &KnowledgeGraphSolver::episode_size);
        def_readonly("batch_size", &KnowledgeGraphSolver::batch_size);
        def_readonly("margin", &KnowledgeGraphSolver::margin);
        def_readonly("l3_regularization", &KnowledgeGraphSolver::l3_regularization);
        def_readonly("adversarial_temperature", &KnowledgeGraphSolver::adversarial_temperature);
        def_readonly("positive_reuse", &KnowledgeGraphSolver::positive_reuse);
        def_readonly("log_frequency", &KnowledgeGraphSolver::log_frequency);
        def_readonly("num_worker", &KnowledgeGraphSolver::num_worker);
        def_readonly("num_sampler", &KnowledgeGraphSolver::num_sampler);
        def_readonly("gpu_memory_limit", &KnowledgeGraphSolver::gpu_memory_limit);
        def_readonly("gpu_memory_cost", &KnowledgeGraphSolver::gpu_memory_cost);

        // numpy views (mutable but unassignable)
        def_property_readonly("entity_embeddings", numpy_view(&KnowledgeGraphSolver::entity_embeddings),
                              "Entity embeddings (2D numpy view).");
        def_property_readonly("relation_embeddings", numpy_view(&KnowledgeGraphSolver::relation_embeddings),
                              "Relation embeddings (2D numpy view).");

        // member functions
        def(py::init<std::vector<int>, int, size_t>(), py::no_gil(),
            py::arg("device_ids") = std::vector<int>(), py::arg("num_sampler_per_worker") = graphvite::kAuto,
            py::arg("gpu_memory_limit") = graphvite::kAuto);

        def("build", &KnowledgeGraphSolver::build, py::no_gil(),
            py::arg("graph"), py::arg("optimizer") = graphvite::Optimizer(graphvite::kAuto),
            py::arg("num_partition") = graphvite::kAuto, py::arg("num_negative") = 64, py::arg("batch_size") = 100000,
            py::arg("episode_size") = graphvite::kAuto,
            "build(graph, optimizer=auto, num_partition=auto, num_negative=64, batch_size=100000, episode_size=auto)"
            R"(
            Determine and allocate all resources for the solver.

            Parameters:
                graph (KnowledgeGraph): knowledge graph
                optimizer (Optimizer or float, optional): optimizer or learning rate
                num_partition (int, optional): number of partitions
                num_negative (int, optional): number of negative samples per positive sample
                batch_size (int, optional): batch size of samples in CPU-GPU transfer
                episode_size (int, optional): number of batches in a partition block
            )");

        def("train", &KnowledgeGraphSolver::train, py::no_gil(),
            py::arg("model") = "RotatE", py::arg("num_epoch") = 2000, py::arg("resume") = false,
            py::arg("relation_lr_multiplier") = 1, py::arg("margin") = 12, py::arg("l3_regularization") = 2e-3,
            py::arg("sample_batch_size") = 2000, py::arg("positive_reuse") = 1, py::arg("adversarial_temperature") = 2,
            py::arg("log_frequency") = 100,
            "train(model='RotatE', num_epoch=2000, resume=False, relation_lr_multiplier=1, margin=12, "
                   "l3_regularization=2e-3, sample_batch_size=2000, positive_reuse=1, adversarial_temperature=2, "
                   "log_frequency=100)"
            R"(
            Train knowledge graph embeddings.

            Parameters:
                model (str, optional): 'TransE', 'DistMult', 'ComplEx', 'SimplE', 'RotatE' or 'QuatE'
                num_epoch (int, optional): number of epochs, i.e. #positive edges / \|E\|
                resume (bool, optional): resume training from learned embeddings or not
                relation_lr_multiplier (float, optional): learning rate multiplier for relation embeddings
                margin (float, optional): logit margin (for TransE & RotatE)
                l3_regularization (float, optional): L3 regularization (for DistMult, ComplEx, SimplE & QuatE)
                sample_batch_size (int, optional): batch size of samples in samplers
                positive_reuse (int, optional): times of reusing positive samples
                adversarial_temperature (float, optional): temperature of self-adversarial negative sampling,
                    disabled when set to non-positive value
                log_frequency (int, optional): log every log_frequency batches
            )");

        def("predict", &KnowledgeGraphSolver::predict_numpy, py::no_gil(),
            py::arg("samples"),
            "predict(samples)"
            R"(
            Predict logits for samples.

            Parameters:
                samples (ndarray): triplets with shape (?, 3), each triplet is ordered as (h, t, r)
            )");

        def("clear", &KnowledgeGraphSolver::clear, py::no_gil(),
            "clear()"
            R"(
            Free CPU and GPU memory, except the embeddings on CPU.
            )");

        def("__repr__", &KnowledgeGraphSolver::info, py::no_gil());
    }
};

template<size_t dim, class Float = float, class Index = size_t>
class pyVisualizationSolver : public py::class_<graphvite::VisualizationSolver<dim, Float, Index>> {
public:
    typedef graphvite::VisualizationSolver<dim, Float, Index> VisualizationSolver;
    typedef py::class_ <VisualizationSolver> Base;
    using Base::attr;
    using Base::def;
    using Base::def_readonly;
    using Base::def_property_readonly;

    template<class... Args>
    pyVisualizationSolver(py::handle scope, const char *name, const Args &...args) :
            Base(scope, signature(name, dim, typeid(Float).name(), typeid(Index).name()).c_str(), args...) {
        attr("__doc__") = "VisualizationSolver(dim, float_type=dtype.float32, index_type=dtype.uint32, "
                                              "device_ids=[], num_sampler_per_worker=auto, gpu_memory_limit=auto)"
        R"(
        Visualization solver.

        Parameters:
            dim (int): dimension of embeddings
            float_type (dtype): type of parameters
            index_type (dtype): type of node indexes
            device_ids (list of int, optional): GPU ids, [] for auto
            num_sampler_per_worker (int, optional): number of sampler thread per GPU
            gpu_memory_limit (int, optional): memory limit for each GPU in bytes
        )";

        // override instance name with template name
        attr("__name__") = py::internal_str(name);
        attr("__qualname__") = py::internal_str(name);

        // data members
        def_readonly("num_partition", &VisualizationSolver::num_partition);
        def_readonly("num_negative", &VisualizationSolver::num_negative);
        def_readonly("sample_batch_size", &VisualizationSolver::sample_batch_size);
        def_readonly("optimizer", &VisualizationSolver::optimizer);
        def_readonly("negative_sample_exponent", &VisualizationSolver::negative_sample_exponent);
        def_readonly("negative_weight", &VisualizationSolver::negative_weight);
        def_readonly("model", &VisualizationSolver::model);
        def_readonly("num_epoch", &VisualizationSolver::num_epoch);
        def_readonly("resume", &VisualizationSolver::resume);
        def_readonly("episode_size", &VisualizationSolver::episode_size);
        def_readonly("batch_size", &VisualizationSolver::batch_size);
        def_readonly("positive_reuse", &VisualizationSolver::positive_reuse);
        def_readonly("log_frequency", &VisualizationSolver::log_frequency);
        def_readonly("num_worker", &VisualizationSolver::num_worker);
        def_readonly("num_sampler", &VisualizationSolver::num_sampler);
        def_readonly("gpu_memory_limit", &VisualizationSolver::gpu_memory_limit);
        def_readonly("gpu_memory_cost", &VisualizationSolver::gpu_memory_cost);

        // numpy views (mutable but unassignable)
        def_property_readonly("coordinates", numpy_view(&VisualizationSolver::coordinates),
                              "Low-dimensional coordinates (2D numpy view).");

        // member functions
        def(py::init<std::vector<int>, int, size_t> (), py::no_gil(),
            py::arg("device_ids") = std::vector<int>(), py::arg("num_sampler_per_worker") = graphvite::kAuto,
            py::arg("gpu_memory_limit") = graphvite::kAuto);

        def("build", &VisualizationSolver::build, py::no_gil(),
            py::arg("graph"), py::arg("optimizer") = graphvite::Optimizer(graphvite::kAuto),
            py::arg("num_partition") = graphvite::kAuto, py::arg("num_negative") = 5, py::arg("batch_size") = 100000,
            py::arg("episode_size") = graphvite::kAuto,
            "build(graph, optimizer=auto, num_partition=auto, num_negative=5, batch_size=100000, episode_size=auto)"
            R"(
            Determine and allocate all resources for the solver.

            Parameters:
                graph (KNNGraph): KNNGraph
                optimizer (Optimizer or float, optional): optimizer or learning rate
                num_partition (int, optional): number of partitions
                num_negative (int, optional): number of negative samples per positive sample
                batch_size (int, optional): batch size of samples in CPU-GPU transfer
                episode_size (int, optional): number of batches in a partition block
            )");

        def("train", &VisualizationSolver::train, py::no_gil(),
            py::arg("model") = "LargeVis", py::arg("num_epoch") = 50, py::arg("resume") = false,
            py::arg("sample_batch_size") = 2000, py::arg("positive_reuse") = 5,
            py::arg("negative_sample_exponent") = 0.75, py::arg("negative_weight") = 3, py::arg("log_frequency") = 1000,
            "train(model='LargeVis', num_epoch=100, resume=False, sample_batch_size=2000, positive_reuse=1, "
                  "negative_sample_exponent=0.75, negative_weight=3, log_frequency=1000)"
            R"(
            Train visualization.

            Parameters:
                model (str, optional): 'LargeVis'
                num_epoch (int, optional): number of epochs, i.e. #positive edges / \|E\|
                resume (bool, optional): resume training from learned embeddings or not
                sample_batch_size (int, optional): batch size of samples in samplers
                positive_reuse (int, optional): times of reusing positive samples
                negative_sample_exponent (float, optional): exponent of degrees in negative sampling
                negative_weight (float, optional): weight for each negative sample
                log_frequency (int, optional): log every log_frequency batches
            )");

        def("clear", &VisualizationSolver::clear, py::no_gil(),
            "clear()"
            R"(
            Free CPU and GPU memory, except the embeddings on CPU.
            )");

        def("__repr__", &VisualizationSolver::info, py::no_gil());
    }
};

template <class... DTypes>
std::function<py::object(const DTypes &..., py::args, py::kwargs)> type_helper(
        const py::module &module, const std::string &name) {
    return [module, name](const DTypes &...dtypes, py::args args, py::kwargs kwargs) {
        auto full_name = signature(name, dtypes...);
        return module.attr(py::str(full_name))(*args, **kwargs);
    };
}

// Optimizer interface
class pyLRSchedule : public py::class_<graphvite::LRSchedule> {
public:
    typedef graphvite::LRSchedule LRSchedule;
    typedef py::class_<LRSchedule> Base;
    using Base::def;
    using Base::def_readonly;

    template<class... Args>
    pyLRSchedule(py::handle scope, const char *name, const Args &...args) :
            Base(scope, name, args...) {
        attr("__doc__") = "LRSchedule(*args, **kwargs)"
        R"(
        Learning Rate Schedule.

        This class has 2 constructors

        .. function:: LRSchedule(type='constant')
        .. function:: LRSchedule(schedule_function)

        Parameters:
            type (str, optional): 'constant' or 'linear'
            schedule_function (callable): function that returns a multiplicative factor,
                given batch id and total number of batches
        )";

        // data members
        def_readonly("type", &LRSchedule::type);
        def_readonly("schedule_function", &LRSchedule::schedule_function);

        // member functions
        def(py::init<std::string>(), py::no_gil(),
            py::arg("type") = "constant");

        def(py::init<LRSchedule::ScheduleFunction>(), py::no_gil(),
            py::arg("schedule_function"));

        py::implicitly_convertible<std::string, graphvite::LRSchedule>();
        py::implicitly_convertible<LRSchedule::ScheduleFunction, graphvite::LRSchedule>();

        def("__repr__", &LRSchedule::info, py::no_gil());
    }
};

class pyOptimizer : public py::class_<graphvite::Optimizer> {
public:
    typedef graphvite::Optimizer Optimizer;
    typedef py::class_<Optimizer> Base;
    using Base::def;
    using Base::def_readonly;

    template<class... Args>
    pyOptimizer(py::handle scope, const char *name, const Args &...args) :
            Base(scope, name, args...) {
        attr("__doc__") = "Optimizer(*args, **kwargs)"
        R"(
        General interface of first-order optimizers.

        This class has 2 constructors

        .. function:: Optimizer(type)
        .. function:: Optimizer(lr=1e-4)

        Parameters:
            type (int): kAuto
            lr (float, optional): initial learning rate
        )";

        // data members
        def_readonly("type", &Optimizer::type);
        def_readonly("lr", &Optimizer::lr);
        def_readonly("weight_decay", &Optimizer::weight_decay);
        def_readonly("schedule", &Optimizer::schedule);

        // member functions
        def(py::init<int>(), py::no_gil(),
            py::arg("type") = graphvite::kAuto);

        def(py::init<float>(), py::no_gil(),
            py::arg("lr") = 1e-4);

        py::implicitly_convertible<int, graphvite::Optimizer>();
        py::implicitly_convertible<float, graphvite::Optimizer>();

        def("__repr__", &Optimizer::info, py::no_gil());
    }
};

class pySGD : public py::class_<graphvite::SGD, graphvite::Optimizer> {
public:
    typedef graphvite::SGD SGD;
    typedef py::class_<SGD, graphvite::Optimizer> Base;
    using Base::def;
    using Base::def_readonly;

    template<class... Args>
    pySGD(py::handle scope, const char *name, const Args &...args) :
            Base(scope, name, args...) {
        attr("__doc__") = "SGD(lr=1e-4, weight_decay=0, schedule='linear')"
        R"(
        Stochastic gradient descent optimizer.

        Parameters:
            lr (float, optional): initial learning rate
            weight_decay (float, optional): weight decay (L2 regularization)
            schedule (str or callable, optional): learning rate schedule
        )";

        // member functions
        def(py::init<float, float, graphvite::LRSchedule>(), py::no_gil(),
            py::arg("lr") = 1e-4, py::arg("weight_decay") = 0,
            py::arg("schedule") = "linear");
    }
};

class pyMomentum : public py::class_<graphvite::Momentum, graphvite::Optimizer> {
public:
    typedef graphvite::Momentum Momentum;
    typedef py::class_<Momentum, graphvite::Optimizer> Base;
    using Base::def_readonly;
    using Base::def;

    template<class... Args>
    pyMomentum(py::handle scope, const char *name, const Args &...args) :
            Base(scope, name, args...) {
        attr("__doc__") = "Momentum(lr=1e-4, weight_decay=0, momentum=0.999, schedule='linear')"
        R"(
        Momentum optimizer.

        Parameters:
            lr (float, optional): initial learning rate
            weight_decay (float, optional): weight decay (L2 regularization)
            momentum (float, optional): momentum coefficient
            schedule (str or callable, optional): learning rate schedule
        )";

        // data members
        def_readonly("momentum", &Momentum::momentum);

        // member functions
        def(py::init<float, float, float, graphvite::LRSchedule>(), py::no_gil(),
            py::arg("lr") = 1e-4, py::arg("weight_decay") = 0, py::arg("momentum") = 0.999,
            py::arg("schedule") = "linear");
    }
};

class pyAdaGrad : public py::class_<graphvite::AdaGrad, graphvite::Optimizer> {
public:
    typedef graphvite::AdaGrad AdaGrad;
    typedef py::class_<AdaGrad, graphvite::Optimizer> Base;
    using Base::def_readonly;
    using Base::def;

    template<class... Args>
    pyAdaGrad(py::handle scope, const char *name, const Args &...args) :
            Base(scope, name, args...) {
        attr("__doc__") = "AdaGrad(lr=1e-4, weight_decay=0, epsilon=1e-10, schedule='linear')"
                R"(
        AdaGrad optimizer.

        Parameters:
            lr (float, optional): initial learning rate
            weight_decay (float, optional): weight decay (L2 regularization)
            epsilon (float, optional): smooth term for numerical stability
            schedule (str or callable, optional): learning rate schedule
        )";

        // data members
        def_readonly("epsilon", &AdaGrad::epsilon);

        // member functions
        def(py::init<float, float, float, graphvite::LRSchedule>(), py::no_gil(),
            py::arg("lr") = 1e-4, py::arg("weight_decay") = 0, py::arg("epsilon") = 1e-10,
            py::arg("schedule") = "linear");
    }
};

class pyRMSprop : public py::class_<graphvite::RMSprop, graphvite::Optimizer> {
public:
    typedef graphvite::RMSprop RMSprop;
    typedef py::class_<RMSprop, graphvite::Optimizer> Base;
    using Base::def_readonly;
    using Base::def;

    template<class... Args>
    pyRMSprop(py::handle scope, const char *name, const Args &...args) :
            Base(scope, name, args...) {
        attr("__doc__") = "RMSprop(lr=1e-4, weight_decay=0, alpha=0.999, epsilon=1e-8, schedule='linear')"
        R"(
        RMSprop optimizer.

        Parameters:
            lr (float, optional): initial learning rate
            weight_decay (float, optional): weight decay (L2 regularization)
            alpha (float, optional): coefficient for moving average of squared gradient
            epsilon (float, optional): smooth term for numerical stability
            schedule (str or callable, optional): learning rate schedule
        )";

        // data members
        def_readonly("alpha", &RMSprop::alpha);
        def_readonly("epsilon", &RMSprop::epsilon);

        // member functions
        def(py::init<float, float, float, float, graphvite::LRSchedule>(), py::no_gil(),
            py::arg("lr") = 1e-4, py::arg("weight_decay") = 0, py::arg("alpha") = 0.999, py::arg("epsilon") = 1e-8,
            py::arg("schedule") = "linear");
    }
};

class pyAdam : public py::class_<graphvite::Adam, graphvite::Optimizer> {
public:
    typedef graphvite::Adam Adam;
    typedef py::class_<Adam, graphvite::Optimizer> Base;
    using Base::def_readonly;
    using Base::def;

    template<class... Args>
    pyAdam(py::handle scope, const char *name, const Args &...args) :
            Base(scope, name, args...) {
        attr("__doc__") = "Adam(lr=1e-4, weight_decay=0, beta1=0.999, beta2=0.99999, epsilon=1e-8, schedule='linear')"
        R"(
        Adam optimizer.

        Parameters:
            lr (float, optional): initial learning rate
            weight_decay (float, optional): weight decay (L2 regularization)
            beta1 (float, optional): coefficient for moving average of gradient
            beta2 (float, optional): coefficient for moving average of squared gradient
            epsilon (float, optional): smooth term for numerical stability
            schedule (str or callable, optional): learning rate schedule
        )";

        // data members
        def_readonly("beta1", &Adam::beta1);
        def_readonly("beta2", &Adam::beta2);
        def_readonly("epsilon", &Adam::epsilon, "");

        // member functions
        def(py::init<float, float, float, float, float, graphvite::LRSchedule>(), py::no_gil(),
            py::arg("lr") = 1e-4, py::arg("weight_decay") = 0, py::arg("beta1") = 0.999, py::arg("beta2") = 0.99999,
            py::arg("epsilon") = 1e-8, py::arg("schedule") = "linear");
    }
};

std::function<py::object(const std::string &, py::args, py::kwargs)> optimizer_helper(const py::module &module) {
    return [module](const std::string &type, py::args args, py::kwargs kwargs) {
        return module.attr(py::str(type))(*args, **kwargs);
    };
}