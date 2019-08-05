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

#include <set>
#include <atomic>
#include <thread>
#include <random>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <curand.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "base/memory.h"
#include "base/vector.h"
#include "base/alias_table.cuh"
#include "core/optimizer.h"

#include "util/common.h"
#include "util/gpu.cuh"
#include "util/debug.h"
#include "util/time.h"

namespace graphvite {

std::mt19937 seed;
const int kMaxPartition = 16;
const int kRandBatchSize = 5e6;
const int kMinBatchSize = 1e4;
const int kMaxNegativeWeight = 10;
const int kSamplePerVertex = 175;
const int kSamplePerVertexWithGlobal = 50;
const int kMinEpisodeSample = 2e7;

// Protocols for scheduling embeddings
typedef unsigned int Protocol;
const Protocol kGlobal = 0x1;
const Protocol kHeadPartition = 0x2;
const Protocol kTailPartition = 0x4;
const Protocol kInPlace = 0x8;
const Protocol kSharedWithPredecessor = 0x10;

/**
 * @brief General interface of graph embedding solvers
 * @tparam _dim dimension of embeddings
 * @tparam _Float floating type of parameters
 * @tparam _Index integral type of node indexes
 * @tparam _Graph type of graph
 * @tparam _Sampler type of edge sampler
 * @tparam _Worker type of training worker
 *
 * @note To add a new solver, you need to
 * - derive a template solver class from SolverMixin
 * - implement all virtual functions for that class
 * - add python binding of instantiations of that class in extension.h & extension.cu
 */
template<size_t _dim, class _Float, class _Index, template<class> class _Graph, template<class> class _Sampler,
        template<class> class _Worker>
class SolverMixin {
public:
    static const size_t dim = _dim;
    typedef _Float Float;
    typedef _Index Index;

    typedef _Graph<Index> Graph;
    typedef _Sampler<SolverMixin> Sampler;
    typedef _Worker<SolverMixin> Worker;

    typedef Vector<dim, Float> Vector;
    typedef typename Sampler::EdgeSample EdgeSample;
    typedef std::function<void(Sampler *, int, int)> SampleFunction;

    Graph *graph = nullptr;
    Index num_vertex;
    size_t num_edge;
    int num_moment;
    int num_embedding, num_partition, num_negative, sample_batch_size;
    float negative_sample_exponent, negative_weight;
    int num_epoch, episode_size, batch_size, positive_reuse;
    int log_frequency;
    bool shuffle_partition, naive_parallel, resume;
    int assignment_offset;
    std::vector<std::shared_ptr<std::vector<Vector>>> embeddings;
    std::vector<std::shared_ptr<std::vector<std::vector<Vector>>>> moments;
    std::vector<Protocol> protocols;
    Protocol sampler_protocol;
    bool tied_weights;
    std::vector<std::vector<Index>> head_partitions, tail_partitions;
    std::vector<std::pair<int, Index>> head_locations, tail_locations;
    AliasTable<Float, size_t> edge_table;
    std::vector<std::vector<std::vector<std::vector<EdgeSample>>>> sample_pools;
    int pool_id = 0;
    std::vector<Sampler *> samplers;
    std::vector<Worker *> workers;
    std::string model;
    std::set<std::string> available_models;
    Optimizer optimizer;
    int num_worker, num_sampler, num_thread;
    size_t gpu_memory_limit, gpu_memory_cost;
    volatile std::atomic<int> batch_id;
    int num_batch;

#define USING_SOLVER_MIXIN(type) \
    using type::dim; \
    using typename type::Float; \
    using typename type::Index; \
    using typename type::Graph; \
    using typename type::Sampler; \
    using typename type::Worker; \
    using typename type::Vector; \
    using typename type::EdgeSample; \
    using typename type::SampleFunction; \
    using type::num_moment; \
    using type::graph; \
    using type::num_vertex; \
    using type::num_edge; \
    using type::num_embedding; \
    using type::num_partition; \
    using type::num_negative; \
    using type::sample_batch_size; \
    using type::positive_reuse; \
    using type::negative_sample_exponent; \
    using type::negative_weight; \
    using type::num_epoch; \
    using type::resume; \
    using type::episode_size; \
    using type::batch_size; \
    using type::optimizer; \
    using type::num_worker; \
    using type::num_sampler; \
    using type::num_thread; \
    using type::edge_table; \
    using type::embeddings; \
    using type::model

    /**
     * @brief Construct a general solver
     * @param device_ids list of GPU ids, {} for auto
     * @param _num_sampler_per_worker number of sampler thread per GPU
     * @param _gpu_memory_limit memory limit for each GPU
     */
    SolverMixin(std::vector<int> device_ids = {}, int num_sampler_per_worker = kAuto, size_t _gpu_memory_limit = kAuto) :
            gpu_memory_limit(_gpu_memory_limit), edge_table(-1), batch_id(0) {
        if (device_ids.empty()) {
            CUDA_CHECK(cudaGetDeviceCount(&num_worker));
            CHECK(num_worker) << "No GPU devices found";
            for (int i = 0; i < num_worker; i++)
                device_ids.push_back(i);
        } else
            num_worker = device_ids.size();
        if (num_sampler_per_worker == kAuto)
            num_sampler_per_worker = std::thread::hardware_concurrency() / num_worker - 1;
        num_sampler = num_sampler_per_worker * num_worker;
        num_thread = num_worker + num_sampler;
        LOG_IF(WARNING, num_sampler > std::thread::hardware_concurrency())
                << "#CPU threads is beyond the hardware concurrency";
        if (gpu_memory_limit == kAuto) {
            gpu_memory_limit = GiB(32);
            size_t free_memory;
            for (auto &&device_id : device_ids) {
                CUDA_CHECK(cudaSetDevice(device_id));
                CUDA_CHECK(cudaMemGetInfo(&free_memory, nullptr));
                gpu_memory_limit = std::min(gpu_memory_limit, free_memory);
            }
        }
        for (int i = 0; i < num_sampler_per_worker; i++)
            for (auto &&device_id : device_ids)
                samplers.push_back(new Sampler(this, device_id));
        for (auto &&device_id : device_ids)
            workers.push_back(new Worker(this, device_id));
    }

    SolverMixin(const SolverMixin &) = delete;

    SolverMixin &operator=(const SolverMixin &) = delete;

    ~SolverMixin() {
        for (auto &&sampler : samplers)
            delete sampler;
        for (auto &&worker : workers)
            delete worker;
    }

    /** Should return the protocols of embeddings */
    virtual std::vector<Protocol> get_protocols() const = 0;

    /** Should return the protocol of negative sampling */
    virtual Protocol get_sampler_protocol() const = 0;

    /** Should return the shapes of embeddings */
    virtual std::vector<Index> get_shapes() const = 0;

    /** Should return all available models of the solver */
    virtual std::set<std::string> get_available_models() const = 0;

    /** Should return the default optimizer type and its hyperparameters */
    virtual Optimizer get_default_optimizer() const = 0;

    /** Should build alias reference for embeddings */
    virtual inline void build_alias() {}

    /** Should initialize embeddings */
    virtual void init_embeddings() = 0;

    /** Should initialize moments */
    virtual void init_moments() {
        for (int i = 0; i < num_embedding; i++) {
            if (protocols[i] & kSharedWithPredecessor)
                continue;
            for (int j = 0; j < num_moment; j++)
                for (auto &&moment : (*moments[i])[j])
                    moment = 0;
        }
    }

    /** Determine and prepare the sampling function */
    virtual SampleFunction get_sample_function() {
        graph->flatten();
        edge_table.build(graph->edge_weights);
        if (naive_parallel)
            return &Sampler::naive_sample;
        else
            return &Sampler::sample;
    }

    /** Determine the minimum number of partitions */
    virtual int get_min_partition() const {
        if (num_worker == 1)
            return num_worker;
        if (tied_weights)
            return num_worker * 2;
        else
            return num_worker;
    }

    /**
     * @brief Determine and allocate all resources for the solver
     * @param _graph graph
     * @param _optimizer optimizer or learning rate
     * @param _num_partition number of partitions
     * @param _num_negative number of negative samples per positive sample
     * @param _batch_size batch size of samples in CPU-GPU transfer
     * @param _episode_size number of batches in a partition block
     */
    void build(Graph &_graph, const Optimizer &_optimizer = kAuto, int _num_partition = kAuto, int _num_negative = 1,
               int _batch_size = 100000, int _episode_size = kAuto) {
        graph = &_graph;
        optimizer = _optimizer;
        if (optimizer.type == "Default") {
            Optimizer default_optimizer = get_default_optimizer();
            if (optimizer.init_lr > 0)
                default_optimizer.init_lr = optimizer.init_lr;
            optimizer = default_optimizer;
        }
        num_vertex = graph->num_vertex;
        num_edge = graph->num_edge;
        num_moment = optimizer.num_moment;
        num_partition = _num_partition;
        num_negative = _num_negative;
        batch_size = _batch_size;
        LOG_IF(WARNING, batch_size < kMinBatchSize)
                << "It is recommended to a minimum batch size of " << kMinBatchSize
                << ", but " << batch_size << " is specified";
        episode_size = _episode_size;
        available_models = get_available_models();
        batch_id = 0;

        // build embeddings & moments
        protocols = get_protocols();
        sampler_protocol = get_sampler_protocol();
        num_embedding = protocols.size();
        embeddings.resize(num_embedding);
        moments.resize(num_embedding);

        auto shapes = get_shapes();
        CHECK(shapes.size() == num_embedding) << "The number of shapes must equal to the number of embedding matrices";
        Protocol all = 0;
        tied_weights = false;
        for (int i = 0; i < num_embedding; i++) {
            Protocol protocol = protocols[i];
            CHECK(bool(protocol & kGlobal) + bool(protocol & kHeadPartition) + bool(protocol & kTailPartition) == 1)
                    << "The embedding matrix can be only binded to either global range, head partition "
                    << "or tail partition";
            if (protocol & (kHeadPartition | kTailPartition)) {
                if (shapes[i] == kAuto)
                    shapes[i] = num_vertex;
                else
                    CHECK(shapes[i] == num_vertex)
                            << "The shape for a partitioned embedding matrix must be `graph->num_vertex`";
            } else
                CHECK(!(protocol & kInPlace)) << "Global embedding matrix can't take in-place update";
            CHECK(shapes[i] != kAuto) << "Can't deduce shape for the " << i << "-th embedding matrix";

            if (protocol & kSharedWithPredecessor) {
                CHECK(i > 0) << "The first embedding matrix can't be shared";
                CHECK(shapes[i] == shapes[i - 1])
                        << "The " << i - 1 << "-th and the " << i << "-th matrices are shared, "
                        << "but different shapes are specified";
                tied_weights = tied_weights || ((protocols[i] | protocols[i - 1]) &
                                                (kHeadPartition | kTailPartition)) == (kHeadPartition | kTailPartition);
                embeddings[i] = embeddings[i - 1];
                moments[i] = moments[i - 1];
            } else {
                embeddings[i] = std::make_shared<std::vector<Vector>>();
                embeddings[i]->resize(shapes[i]);
                moments[i] = std::make_shared<std::vector<std::vector<Vector>>>(num_moment);
                for (int j = 0; j < num_moment; j++)
                    (*moments[i])[j].resize(shapes[i]);
            }
            all |= protocol;
        }
        if ((all & kHeadPartition) != (all & kTailPartition)) {
            LOG_IF(WARNING, !(all & kHeadPartition))
                    << "No embedding matrix is binded to head partition";
            LOG_IF(WARNING, !(all & kTailPartition))
                    << "No embedding matrix is binded to tail partition";
        }
        CHECK(bool(sampler_protocol & kGlobal) + bool(sampler_protocol & (kHeadPartition | kTailPartition)) == 1)
                << "The negative sampler can't be binded to global range and any partition at the same time";

        build_alias();

        auto min_partition = get_min_partition();
        if (num_partition == kAuto) {
            for (num_partition = min_partition; num_partition < kMaxPartition; num_partition += min_partition)
                if (gpu_memory_demand(protocols, shapes, sampler_protocol, num_moment, num_partition, num_negative,
                                      batch_size) < gpu_memory_limit)
                    break;
        } else {
            CHECK(num_partition >= min_partition) << "#partition should be no less than " << min_partition;
            LOG_IF(WARNING, num_partition > kMaxPartition)
                    << "It is recommended to use a maximum #partition of " << kMaxPartition
                    << ", but " << num_partition << " partitions are specified";
        }
        gpu_memory_cost = gpu_memory_demand(
                protocols, shapes, sampler_protocol, num_moment, num_partition, num_negative, batch_size);
        CHECK(gpu_memory_cost < gpu_memory_limit)
                << "Can't satisfy the specified GPU memory limit";
        if (episode_size == kAuto) {
            if (all & kGlobal)
                episode_size = float(num_vertex * kSamplePerVertexWithGlobal) / num_partition / batch_size;
            else
                episode_size = float(num_vertex * kSamplePerVertex) / num_partition / batch_size;
            episode_size = std::max(episode_size, 1);
            if (num_partition == 1)
                // for single partition, we don't need to use very small episode size
                episode_size = std::max(episode_size, kMinEpisodeSample / batch_size);
        }

        // use naive data parallel if there is no partition
        naive_parallel = !(all & (kHeadPartition | kTailPartition));
        // use shuffle partition to get unbiased moment estimation for global bindings
        shuffle_partition = !naive_parallel && (all & kGlobal) && num_moment > 0;
        assignment_offset = 0;

        if (!naive_parallel) {
            head_partitions = partition(graph->vertex_weights, num_partition);
            tail_partitions = partition(graph->vertex_weights, num_partition);
            head_locations.resize(num_vertex);
            tail_locations.resize(num_vertex);
            for (int i = 0; i < num_partition; i++)
                for (Index j = 0; j < head_partitions[i].size(); j++) {
                    Index head_id = head_partitions[i][j];
                    head_locations[head_id] = {i, j};
                }
            for (int i = 0; i < num_partition; i++)
                for (Index j = 0; j < tail_partitions[i].size(); j++) {
                    Index tail_id = tail_partitions[i][j];
                    tail_locations[tail_id] = {i, j};
                }
        }

        // allocating sample pool is slow
        sample_pools.resize(2);
        for (auto &&sample_pool : sample_pools) {
            sample_pool.resize(num_partition);
            for (auto &&partition_pool : sample_pool) {
                if (naive_parallel)
                    partition_pool.resize(1);
                else
                    partition_pool.resize(num_partition);
                for (auto &&pool : partition_pool)
                    pool.resize(episode_size * batch_size);
            }
        }
        for (auto &&sampler : samplers)
            sampler->build();
        for (auto &&worker : workers)
            worker->build();
    }

    virtual inline std::string name() const {
        std::stringstream ss;
        ss << "SolverMixin<" << dim << ", "
           << pretty::type2name<Float>() << ", " << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    virtual inline std::string resource_info() const {
        std::stringstream ss;
        ss << "#worker: " << num_worker << ", #sampler: " << num_sampler;
        if (naive_parallel)
            ss << ", naive parallel" << std::endl;
        else
            ss << ", #partition: " << num_partition << std::endl;
        ss << "tied weights: " << pretty::yes_no(tied_weights) << ", episode size: " << episode_size << std::endl;
        ss << "gpu memory limit: " << pretty::size_string(gpu_memory_limit) << std::endl;
        ss << "gpu memory cost: " << pretty::size_string(gpu_memory_cost);
        return ss.str();
    }

    virtual inline std::string sampling_info() const {
        std::stringstream ss;
        ss << "positive sample batch size: " << sample_batch_size << std::endl;
        ss << "#negative: " << num_negative << ", negative sample exponent: " << negative_sample_exponent;
        return ss.str();
    }

    virtual inline std::string training_info() const {
        std::stringstream ss;
        ss << "model: " << model << std::endl;
        ss << optimizer.info() << std::endl;
        ss << "#epoch: " << num_epoch << ", batch size: " << batch_size << std::endl;
        ss << "resume: " << pretty::yes_no(resume) << std::endl;
        ss << "positive reuse: " << positive_reuse << ", negative weight: " << negative_weight;
        return ss.str();
    }

    /** Return information about the solver */
    std::string info() const {
        std::stringstream ss;
        ss << name() << std::endl;
        ss << pretty::header("Resource") << std::endl;
        ss << resource_info() << std::endl;
        ss << pretty::header("Sampling") << std::endl;
        ss << sampling_info() << std::endl;
        ss << pretty::header("Training") << std::endl;
        ss << training_info();
        return ss.str();
    }

    /** Determine the schedule of partitions */
    virtual std::vector<std::vector<std::pair<int, int>>> get_schedule() const {
        std::vector<std::vector<std::pair<int, int>>> schedule;
        std::vector<std::pair<int, int>> assignment(num_worker);

        if (num_partition == 1)
            return {{{0, 0}}};

        if (naive_parallel) {
            for (int i = 0; i < num_worker; i++)
                assignment[i] = {i, 0};
            return {assignment};
        }

        if (tied_weights)
            for (int x = 0; x < num_partition; x += num_worker * 2)
                for (int y = 0; y < num_partition; y += num_worker * 2) {
                    // diagonal
                    for (int i = 0; i < num_worker; i++) {
                        int head_partition_id = x + i;
                        int tail_partition_id = y + i;
                        assignment[i] = {head_partition_id, tail_partition_id};
                    }
                    schedule.push_back(assignment);
                    for (int i = 0; i < num_worker; i++) {
                        int head_partition_id = x + num_worker + i;
                        int tail_partition_id = y + num_worker + i;
                        assignment[i] = {head_partition_id, tail_partition_id};
                    }
                    schedule.push_back(assignment);
                    for (int group_size = 1; group_size <= num_worker; group_size *= 2)
                        for (int offset = 0; offset < group_size; offset++) {
                            for (int i = 0; i < num_worker; i++) {
                                int head_partition_id = x + (i / group_size * 2) * group_size + i % group_size;
                                int tail_partition_id = y + (i / group_size * 2 + 1) * group_size
                                                        + (i + offset) % group_size;
                                assignment[i] = {head_partition_id, tail_partition_id};
                            }
                            schedule.push_back(assignment);
                            for (int i = 0; i < num_worker; i++)
                                std::swap(assignment[i].first, assignment[i].second);
                            schedule.push_back(assignment);
                        }
                }
        else
            for (int x = 0; x < num_partition; x += num_worker)
                for (int y = 0; y < num_partition; y += num_worker)
                    for (int offset = 0; offset < num_worker; offset++) {
                        for (int i = 0; i < num_worker; i++) {
                            int head_partition_id = x + (i + offset) % num_worker;
                            int tail_partition_id = y + i;
                            assignment[i] = {head_partition_id, tail_partition_id};
                        }
                        schedule.push_back(assignment);
                    }

        return schedule;
    }

    /**
     * @brief Train graph embeddings
     * @param _model model
     * @param _num_epoch number of epochs, i.e. #positive edges / |E|
     * @param _resume resume training from learned embeddings or not
     * @param _sample_batch_size batch size of samples in samplers
     * @param _positive_reuse times of reusing positive samples
     * @param _negative_sample_exponent exponent of degrees in negative sampling
     * @param _negative_weight weight for each negative sample
     * @param _log_frequency log every log_frequency batches
     */
    void train(const std::string &_model, int _num_epoch = 2000, bool _resume = false, int _sample_batch_size = 2000,
               int _positive_reuse = 1, float _negative_sample_exponent = 0.75, float _negative_weight = 5,
               int _log_frequency = 1000) {
        CHECK(graph) << "The model must be built on a graph first";
        model = _model;
        CHECK(available_models.find(model) != available_models.end()) << "Invalid model `" << model << "`";
        num_epoch = _num_epoch;
        resume = _resume;
        sample_batch_size = _sample_batch_size;
        positive_reuse = _positive_reuse;
        negative_sample_exponent = _negative_sample_exponent;
        negative_weight = _negative_weight;
        LOG_IF(WARNING, negative_weight > kMaxNegativeWeight)
                << "It is recommended to a maximum negative weight of " << kMaxNegativeWeight
                << ", but " << negative_weight << " is specified";
        log_frequency = _log_frequency;

        LOG(WARNING) << pretty::block(info());
        if (!resume) {
            init_embeddings();
            init_moments();
            batch_id = 0;
        }
        num_batch = batch_id + num_epoch * num_edge / batch_size;

        std::vector<std::thread> sample_threads(num_sampler);
        std::vector<std::thread> worker_threads(num_worker);
        int num_sample = episode_size * batch_size;
        int work_load = (num_sample + num_sampler - 1) / num_sampler;
        auto schedule = get_schedule();

        SampleFunction sample_function = get_sample_function();
        {
            Timer timer("Sample threads");
            for (int i = 0; i < num_sampler; i++)
                sample_threads[i] = std::thread(sample_function, samplers[i], work_load * i,
                                                std::min(work_load * (i + 1), num_sample));
            for (auto &&thread : sample_threads)
                thread.join();
        }
        while (batch_id < num_batch) {
            pool_id ^= 1;
            if (shuffle_partition)
                assignment_offset = (assignment_offset + 1) % num_partition;
            for (int i = 0; i < num_sampler; i++)
                sample_threads[i] = std::thread(sample_function, samplers[i], work_load * i,
                                                std::min(work_load * (i + 1), num_sample));
            for (auto &&assignment : schedule) {
                for (int i = 0; i < assignment.size(); i++)
                    worker_threads[i] = std::thread(&Worker::train, workers[i],
                                                    assignment[i].first,
                                                    (assignment[i].second + assignment_offset) % num_partition);
                for (int i = 0; i < assignment.size(); i++)
                    worker_threads[i].join();
            }
            {
                Timer timer("Wait for sample threads");
                for (auto &&thread : sample_threads)
                    thread.join();
            }
        }
        for (int i = 0; i < num_worker; i++)
            worker_threads[i] = std::thread(&Worker::write_back, workers[i]);
        for (auto &&thread : worker_threads)
            thread.join();
    }

    /** Free CPU and GPU memory, except the embeddings on CPU */
    void clear() {
        decltype(moments)().swap(moments);
        decltype(head_partitions)().swap(head_partitions);
        decltype(tail_partitions)().swap(tail_partitions);
        decltype(sample_pools)().swap(sample_pools);
        for (auto &&sampler : samplers)
            sampler->clear();
        for (auto &&worker : workers)
            worker->clear();
    }

    /**
     * @param protocols protocols of embeddings
     * @param shapes shapes of embeddings
     * @param sampler_protocol protocol of negative sampling
     * @param num_moment number of moment statistics in optimizers
     * @param num_partition numebr of partitions
     * @param num_negative number of negative samples per postive sample
     * @param batch_size batch size of samples in CPU-GPU transfer
     * @return GPU memory cost
     */
    static size_t gpu_memory_demand(const std::vector<Protocol> &protocols, const std::vector<Index> &shapes,
                                    Protocol sampler_protocol = kTailPartition, int num_moment = 0,
                                    int num_partition = 4, int num_negative = 1, int batch_size = 100000) {
        auto partition_shapes = shapes;
        int num_embedding = protocols.size();
        Index num_vertex;
        for (int i = 0; i < num_embedding; i++) {
            Protocol protocol = protocols[i];
            if (protocol & (kHeadPartition | kTailPartition)) {
                num_vertex = shapes[i];
                partition_shapes[i] = (shapes[i] + num_partition - 1) / num_partition;
            }
            if (protocol & kSharedWithPredecessor) {
                if (num_partition == 1 || (protocol & kGlobal))
                    partition_shapes[i] = 0;
                if ((protocol & (kHeadPartition | kTailPartition)) ==
                    (protocols[i - 1] & (kHeadPartition | kTailPartition)))
                    partition_shapes[i] = 0;
            }
        }
        Index sampler_size = 0;
        if (sampler_protocol | kHeadPartition)
            sampler_size += (num_vertex + num_partition - 1) / num_partition;
        if (sampler_protocol | kTailPartition)
            sampler_size += (num_vertex + num_partition - 1) / num_partition;
        if (sampler_protocol | kGlobal)
            sampler_size += num_vertex;

        size_t demand = 0;
        demand += Sampler::gpu_memory_demand();
        demand += Worker::gpu_memory_demand(protocols, partition_shapes, sampler_size, num_moment, num_negative,
                                            batch_size);
        return demand;
    }

private:
    /**
     * @brief Generate partition for nodes s.t. each partition has similar sum of weights.
     * @param weights degrees of nodes
     * @param num_partition number of partitions
     * @return partitioned indexes
     */
    static std::vector<std::vector<Index>> partition(const std::vector<Float> &weights, int num_partition) {
        std::vector<Index> indexes(weights.size());
        for (Index i = 0; i < indexes.size(); i++)
            indexes[i] = i;
        std::sort(indexes.begin(), indexes.end(),
                  [&weights](Index x, Index y) { return weights[x] > weights[y]; });
        std::vector<std::vector<Index>> parts(num_partition);
        for (Index i = 0; i < indexes.size(); i++) {
            Index index = indexes[i];
            int part_id = i % (num_partition * 2);
            part_id = std::min(part_id, num_partition * 2 - 1 - part_id);
            parts[part_id].push_back(index);
        }
        return parts;
    }
};

/**
 * @brief General interface of edge samplers
 * @tparam _Solver type of graph embedding solver
 * @tparam _Attributes types of additional edge attributes
 *
 * @note To add a new sampler, you need to
 * - derive a template sampler class from SamplerMixin
 * - implement all virtual functions for that class
 * - bind that class to a solver as a template parameter
 */
template<class _Solver, class ..._Attributes>
class SamplerMixin {
public:
    typedef _Solver Solver;
    typedef typename Solver::Float Float;
    typedef typename Solver::Index Index;

    typedef std::tuple<_Attributes...> Attributes;
    typedef std::tuple<Index, Index, _Attributes...> EdgeSample;
    typedef typename Solver::Graph::Edge Edge;

    Solver *solver;
    int device_id;
    cudaStream_t stream;
    Memory<double, int> random;
    curandGenerator_t generator;
    int pool_size;

#define USING_SAMPLER_MIXIN(type) \
    using typename type::Solver; \
    using typename type::Float; \
    using typename type::Index; \
    using typename type::Attributes; \
    using typename type::Edge; \
    using type::solver; \
    using type::device_id; \
    using type::stream; \
    using type::random; \
    using type::generator; \
    using type::pool_size

    /**
     * Construct a general edge sampler
     * @param _solver pointer to graph embedding solver
     * @param _device_id GPU id
     */
    SamplerMixin(Solver *_solver, int _device_id) :
            solver(_solver), device_id(_device_id), random(device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));

        CUDA_CHECK(cudaStreamCreate(&stream));

        random.stream = stream;
        CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        std::uniform_int_distribution<unsigned long long> random_seed(0, ULLONG_MAX);
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, random_seed(seed)));
        CURAND_CHECK(curandSetStream(generator, stream));
    }

    /** Should return the additional attributes */
    virtual Attributes get_attributes(const Edge &edge) const = 0;

    /** Determine and allocate all resources for the sampler */
    void build() {
        CUDA_CHECK(cudaSetDevice(device_id));

        pool_size = solver->sample_pools[0][0][0].size();
        random.resize(kRandBatchSize);
        CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));
    }

    /** Free GPU memory */
    void clear() {
        random.resize(0);
    }

    /** Sample edges for naive parallel. This function can be parallelized. */
    void naive_sample(int start, int end) {
        CUDA_CHECK(cudaSetDevice(device_id));

        random.to_host();
        CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));

        auto &sample_pool = solver->sample_pools[solver->pool_id ^ 1];
        int partition_id = 0, rand_id = 0, offset = start;
        std::vector<Index> heads(solver->sample_batch_size);
        std::vector<Index> tails(solver->sample_batch_size);
        std::vector<Attributes> attributes(solver->sample_batch_size);
        while (partition_id < solver->num_partition) {
            for (int i = 0; i < solver->sample_batch_size; i++) {
                if (rand_id > kRandBatchSize - 2) {
                    random.to_host();
                    CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));
                    rand_id = 0;
                }
                size_t edge_id = solver->edge_table.sample(random[rand_id++], random[rand_id++]);

                heads[i] = std::get<0>(solver->graph->edges[edge_id]);
                tails[i] = std::get<1>(solver->graph->edges[edge_id]);
                attributes[i] = get_attributes(solver->graph->edges[edge_id]);
            }
            for (int i = 0; i < solver->sample_batch_size; i++) {
                auto &pool = sample_pool[partition_id][0];
                pool[offset] = std::tuple_cat(std::tie(heads[i], tails[i]), attributes[i]);
                if (++offset == end) {
                    if (++partition_id == solver->num_partition)
                        return;
                    offset = start;
                }
            }
        }
    }

    /** Sample edges. This function can be parallelized. */
    void sample(int start, int end) {
        CUDA_CHECK(cudaSetDevice(device_id));

        random.to_host();
        CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));

        auto &sample_pool = solver->sample_pools[solver->pool_id ^ 1];
        std::vector<std::vector<int>> offsets(solver->num_partition);
        for (auto &&partition_offsets : offsets)
            partition_offsets.resize(solver->num_partition, start);
        int num_complete = 0, rand_id = 0;
        std::vector<std::pair<int, Index>> heads(solver->sample_batch_size);
        std::vector<std::pair<int, Index>> tails(solver->sample_batch_size);
        std::vector<Attributes> attributes(solver->sample_batch_size);
        while (num_complete < solver->num_partition * solver->num_partition) {
            for (int i = 0; i < solver->sample_batch_size; i++) {
                if (rand_id > kRandBatchSize - 2) {
                    random.to_host();
                    CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, kRandBatchSize));
                    rand_id = 0;
                }
                size_t edge_id = solver->edge_table.sample(random[rand_id++], random[rand_id++]);

                Index head_global_id = std::get<0>(solver->graph->edges[edge_id]);
                Index tail_global_id = std::get<1>(solver->graph->edges[edge_id]);
                heads[i] = solver->head_locations[head_global_id];
                tails[i] = solver->tail_locations[tail_global_id];
                attributes[i] = get_attributes(solver->graph->edges[edge_id]);
            }
            for (int i = 0; i < solver->sample_batch_size; i++) {
                int head_partition_id = heads[i].first;
                int tail_partition_id = tails[i].first;
                int &offset = offsets[head_partition_id][tail_partition_id];
                if (offset < end) {
                    auto &pool = sample_pool[head_partition_id][tail_partition_id];
                    Index head_local_id = heads[i].second;
                    Index tail_local_id = tails[i].second;
                    pool[offset] = std::tuple_cat(std::tie(head_local_id, tail_local_id), attributes[i]);
                    if (++offset == end)
                        num_complete++;
                }
            }
        }
    }

    /** @return GPU memory cost */
    static size_t gpu_memory_demand() {
        size_t demand = 0;
        demand += decltype(random)::gpu_memory_demand(kRandBatchSize);
        return demand;
    }
};

/**
 * General interface for training workers
 * @tparam _Solver type of graph embedding solver
 *
 * @note To add a new worker, you need to
 * - derive a template worker class from WorkerMixin
 * - implement all virtual functions for that class
 * - bind that class to a solver as a template parameter
 */
template<class _Solver>
class WorkerMixin {
public:
    typedef _Solver Solver;
    typedef typename Solver::Float Float;
    typedef typename Solver::Index Index;
    typedef typename Solver::Vector Vector;
    typedef typename Solver::EdgeSample EdgeSample;

    static const int sample_size = sizeof(EdgeSample) / sizeof(Index);
    static_assert(sizeof(EdgeSample) % sizeof(Index) == 0, "sizeof(EdgeSample) must be a multiplier of sizeof(Index)");

    Solver *solver;
    int device_id;
    Optimizer optimizer;
    cudaStream_t work_stream, sample_stream;
    Index head_partition_size, tail_partition_size;
    std::vector<std::shared_ptr<Memory<Vector, Index>>> embeddings, gradients;
    std::vector<std::shared_ptr<std::vector<Memory<Vector, Index>>>> moments;
    int head_partition_id, tail_partition_id;
    std::vector<Index> head_global_ids, tail_global_ids;
    std::vector<Protocol> protocols;
    Protocol sampler_protocol;
    AliasTable<Float, Index> negative_sampler;
    Memory<Index, int> batch, negative_batch;
#ifdef USE_LOSS
    Memory<Float, int> loss;
#endif
    Memory<double, int> random;
    curandGenerator_t generator;
    int num_moment;
    int num_embedding, num_negative, batch_size;
    int log_frequency;

#define USING_WORKER_MIXIN(type) \
    using typename type::Solver; \
    using typename type::Float; \
    using typename type::Index; \
    using typename type::Vector; \
    using type::solver; \
    using type::work_stream; \
    using type::head_partition_size; \
    using type::tail_partition_size; \
    using type::head_partition_id; \
    using type::tail_partition_id; \
    using type::head_global_ids; \
    using type::tail_global_ids; \
    using type::negative_sampler; \
    using type::batch; \
    using type::negative_batch; \
    using type::batch_size; \
    using type::embeddings; \
    using type::moments; \
    using type::gradients; \
    using type::num_moment; \
    using type::optimizer

    /**
     * Construct a general training worker
     * @param _solver pointer to graph embedding solver
     * @param _device_id GPU id
     */
    WorkerMixin(Solver *_solver, int _device_id) :
            solver(_solver), device_id(_device_id), negative_sampler(device_id),
            batch(device_id), negative_batch(device_id),
#ifdef USE_LOSS
            loss(device_id),
#endif
            random(device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));

        CUDA_CHECK(cudaStreamCreate(&work_stream));
        CUDA_CHECK(cudaStreamCreate(&sample_stream));

        // work stream
        batch.stream = work_stream;
        negative_batch.stream = work_stream;
#ifdef USE_LOSS
        loss.stream = work_stream;
#endif
        // sample stream
        negative_sampler.stream = sample_stream;
        random.stream = sample_stream;
        CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        std::uniform_int_distribution<unsigned long long> random_seed(0, ULLONG_MAX);
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, random_seed(seed)));
        CURAND_CHECK(curandSetStream(generator, sample_stream));
    }

    WorkerMixin(const WorkerMixin &) = delete;

    WorkerMixin &operator=(const WorkerMixin &) = delete;

    /** Should call the corresponding GPU kernel */
    virtual bool kernel_dispatch() = 0;

    /** Build the alias table for negative sampling */
    virtual void build_negative_sampler() {
        std::vector <Float> negative_weights;
        if (sampler_protocol & kHeadPartition)
            for (auto &&head_global_id : head_global_ids)
                negative_weights.push_back(std::pow(solver->graph->vertex_weights[head_global_id],
                                                    solver->negative_sample_exponent));
        if (sampler_protocol & kTailPartition)
            for (auto &&tail_global_id : tail_global_ids)
                negative_weights.push_back(std::pow(solver->graph->vertex_weights[tail_global_id],
                                                    solver->negative_sample_exponent));
        if (sampler_protocol & kGlobal)
            for (auto &&vertex_weight : solver->graph->vertex_weights)
                negative_weights.push_back(std::pow(vertex_weight, solver->negative_sample_exponent));
        negative_sampler.build(negative_weights);
    }

    /** Determine and allocate all resources for the worker */
    void build() {
        num_embedding = solver->num_embedding;
        num_moment = solver->num_moment;
        num_negative = solver->num_negative;
        batch_size = solver->batch_size;
        optimizer = solver->optimizer;
        protocols = solver->protocols;
        sampler_protocol = solver->sampler_protocol;

        embeddings.resize(num_embedding);
        gradients.resize(num_embedding);
        moments.resize(num_embedding);
        for (int i = 0; i < num_embedding; i++) {
            embeddings[i] = std::make_shared<Memory<Vector, Index>>(device_id, 0, work_stream);
            gradients[i] = std::make_shared<Memory<Vector, Index>>(device_id, 0, work_stream);
            moments[i] = std::make_shared<std::vector<Memory<Vector, Index>>>();
            for (int j = 0; j < num_moment; j++)
                moments[i]->push_back(Memory<Vector, Index>(device_id, 0, work_stream));
        }

        batch.resize(batch_size * sample_size);
        negative_batch.resize(batch_size * num_negative);
#ifdef USE_LOSS
        loss.resize(batch_size);
#endif
        random.resize(batch_size * num_negative * 2);

        head_partition_id = -1;
        tail_partition_id = -1;
    }

    /** Free GPU memory */
    void clear() {
        embeddings.clear();
        gradients.clear();
        moments.clear();
        batch.resize(0);
        negative_batch.resize(0);
#ifdef USE_LOSS
        loss.resize(0);
#endif
        random.resize(0);
        negative_sampler.clear();
    }

    /** Load an embedding matrix to GPU, along with its gradients and moment statistics */
    void load_partition_one(int id) {
        Protocol protocol = protocols[id];
        if ((protocol & kSharedWithPredecessor) && ((protocol & kGlobal) || head_partition_id == tail_partition_id)) {
            embeddings[id] = embeddings[id - 1];
            gradients[id] = gradients[id - 1];
            moments[id] = moments[id - 1];
            return;
        }
        if (id > 0 && embeddings[id] == embeddings[id - 1]) {
            embeddings[id] = std::make_shared<Memory<Vector, Index>>(device_id, 0, work_stream);
            gradients[id] = std::make_shared<Memory<Vector, Index>>(device_id, 0, work_stream);
            moments[id] = std::make_shared<std::vector<Memory<Vector, Index>>>();
            for (int i = 0; i < num_moment; i++)
                moments[id]->push_back(Memory<Vector, Index>(device_id, 0, work_stream));
        }
        auto &global_embedding = *(solver->embeddings[id]);
        auto &global_moment = *(solver->moments[id]);
        auto &embedding = *embeddings[id];
        auto &moment = *moments[id];
        auto &gradient = *gradients[id];
        std::vector<Index> none, *mapping = &none;

        if (protocol & kHeadPartition)
            mapping = &head_global_ids;
        if (protocol & kTailPartition)
            mapping = &tail_global_ids;

        embedding.gather(global_embedding, *mapping);
        embedding.to_device_async();
        // only load partitioned moments, or global moments if uninitialized
        if (!(protocol & kGlobal) || (num_moment && moment[0].count == 0))
            for (int i = 0; i < num_moment; i++) {
                moment[i].gather(global_moment[i], *mapping);
                moment[i].to_device_async();
            }
        if (!(protocol & kInPlace)) {
            gradient.fill(0, embedding.count);
            gradient.to_device_async();
        }
    }

    /** Write back an embedding matrix from GPU, along with its gradients and moment statistics */
    void write_back_one(int id) {
        Protocol protocol = protocols[id];
        if (id > 0 && embeddings[id] == embeddings[id - 1])
            return;
        auto &global_embedding = *(solver->embeddings[id]);
        auto &global_moment = *(solver->moments[id]);
        auto &embedding = *embeddings[id];
        auto &moment = *moments[id];
        auto &gradient = *gradients[id];
        std::vector<Index> none, *mapping = &none;

        if (protocol & kHeadPartition)
            mapping = &head_global_ids;
        if (protocol & kTailPartition)
            mapping = &tail_global_ids;

        if (protocol & kInPlace) {
            embedding.to_host();
            embedding.scatter(global_embedding, *mapping);
        }
        else {
            gradient.to_host();
            for (Index i = 0; i < gradient.count; i++)
                gradient[i] /= solver->num_worker;
            gradient.scatter_sub(global_embedding, *mapping);
        }
        // only write back partitioned moments
        if (!(protocol & kGlobal))
            for (int i = 0; i < num_moment; i++) {
                moment[i].to_host();
                moment[i].scatter(global_moment[i], *mapping);
            }
    }

    /**
     * @brief Load a partition of the sample pool. Update the cache automatically
     * @param _head_partition_id id of head partition
     * @param _tail_partition_id id of tail partition
     */
    void load_partition(int _head_partition_id, int _tail_partition_id) {
        // check hit/miss
        bool cold_cache = head_partition_id == -1 || tail_partition_id == -1;
        std::vector<bool> hit(num_embedding, false);
        if (!cold_cache) {
            for (int i = 0; i < num_embedding; i++) {
                Protocol protocol = protocols[i];
                if ((protocol & kSharedWithPredecessor) && !(protocol & kGlobal) && !hit[i - 1]) {
                    // check swap hit
                    if (head_partition_id == _tail_partition_id && tail_partition_id == _head_partition_id) {
                        embeddings[i].swap(embeddings[i - 1]);
                        gradients[i].swap(gradients[i - 1]);
                        moments[i].swap(moments[i - 1]);
                        hit[i] = true;
                        hit[i - 1] = true;
                    }
                }
                // if the current embedding is shared with predecessor
                // then it is necessary to have hit on the predecessor
                if (!(i > 0 && embeddings[i] == embeddings[i - 1] && !hit[i - 1])){
                    hit[i] = hit[i] || ((protocol & kHeadPartition) && head_partition_id == _head_partition_id);
                    hit[i] = hit[i] || ((protocol & kTailPartition) && tail_partition_id == _tail_partition_id);
                }
            }
            for (int i = 0; i < num_embedding; i++)
                if (!hit[i])
                    write_back_one(i);
        }
        bool sampler_hit = (sampler_protocol & kGlobal) && !cold_cache;
        sampler_hit = sampler_hit || ((sampler_protocol & kHeadPartition) && (sampler_protocol & kTailPartition)
                                      && head_partition_id == _tail_partition_id
                                      && _head_partition_id == tail_partition_id);
        sampler_hit = sampler_hit || ((sampler_protocol & (kHeadPartition | kTailPartition)) == kHeadPartition
                                      && head_partition_id == _head_partition_id);
        sampler_hit = sampler_hit || ((sampler_protocol & (kHeadPartition | kTailPartition)) == kTailPartition
                                      && tail_partition_id == _tail_partition_id);

        // load cache
        if (head_partition_id != _head_partition_id) {
            head_partition_id = _head_partition_id;
            if (!solver->naive_parallel) {
                head_global_ids = solver->head_partitions[head_partition_id];
                head_partition_size = head_global_ids.size();
            }
        }
        if (tail_partition_id != _tail_partition_id) {
            tail_partition_id = _tail_partition_id;
            if (!solver->naive_parallel) {
                tail_global_ids = solver->tail_partitions[tail_partition_id];
                tail_partition_size = tail_global_ids.size();
            }
        }
        if (!sampler_hit) {
            build_negative_sampler();
            negative_sampler.to_device_async();
        }
        for (int i = 0; i < num_embedding; i++)
            if (!hit[i])
                load_partition_one(i);
    }

    /** Write back all cache */
    void write_back() {
        bool cold_cache = head_partition_id == -1 || tail_partition_id == -1;
        if (cold_cache)
            return;
        for (int i = 0; i < num_embedding; i++)
            write_back_one(i);
    }

    /**
     * @brief Train embeddings with samples in the partition
     * @param _head_partition_id id of head partition
     * @param _tail_partition_id id of tail partition
     */
    void train(int _head_partition_id, int _tail_partition_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
        load_partition(_head_partition_id, _tail_partition_id);

        auto &samples = solver->sample_pools[solver->pool_id][head_partition_id][tail_partition_id];
        log_frequency = solver->log_frequency;
        for (int i = 0; i < solver->positive_reuse; i++)
            for (int j = 0; j < solver->episode_size; j++) {
                memcpy(batch.host_ptr, &samples[j * batch_size], batch_size * sample_size * sizeof(Index));
                train_batch(solver->batch_id++);
            }
    }

    /** Train a single batch */
    virtual void train_batch(int batch_id) {
        Timer batch_timer("Batch", log_frequency);
        if (batch_id % log_frequency == 0)
            LOG(INFO) << "Batch id: " << batch_id << " / " << solver->num_batch;
        batch.to_device_async();
        // Sampling
        {
            Timer timer("Sampling", log_frequency);
            int num_sample = batch_size * num_negative;
            {
                Timer timer("Random", log_frequency);
                CURAND_CHECK(curandGenerateUniformDouble(generator, random.device_ptr, num_sample * 2));
                negative_sampler.device_sample(random, &negative_batch);
            }
            CUDA_CHECK(cudaStreamSynchronize(sample_stream));
        }
#ifdef USE_LOSS
        // Loss (last batch)
        if (batch_id % log_frequency == 0){
            Timer timer("Loss", log_frequency);
            loss.to_host();
            Float batch_loss = 0;
            for (int i = 0; i < batch_size; i++)
                batch_loss += loss[i];
            LOG(INFO) << "loss = " << batch_loss / batch_size;
        }
#endif
        // Train
        {
            Timer timer("Train", log_frequency);
            optimizer.apply_schedule(batch_id, solver->num_batch);
            CHECK(kernel_dispatch())
                    << "Can't find a kernel implementation of `" << solver->model << " with " << optimizer.type;
        }
    }

    /**
     * @param protocols protocols of embeddings
     * @param shapes shapes of embeddings
     * @param sampler_size size of the negative sampling distribution
     * @param num_moment number of moment statistics in optimizers
     * @param num_negative number of negative samples per positive sample
     * @param batch_size batch size of samples in CPU-GPU transfer
     * @return GPU memory cost
     */
    static size_t gpu_memory_demand(const std::vector<Protocol> &protocols, const std::vector<Index> &shapes,
                                    Index sampler_size = 0, int num_moment = 0, int num_negative = 1,
                                    int batch_size = 100000) {
        size_t demand = 0;
        int num_embedding = protocols.size();
        for (int i = 0; i < num_embedding; i++) {
            Protocol protocol = protocols[i];
            demand += Memory<Vector, Index>::gpu_memory_demand(shapes[i]) * (num_moment + 1);
            if (!(protocol & kInPlace))
                demand += Memory<Vector, Index>::gpu_memory_demand(shapes[i]);
        }
        demand += decltype(batch)::gpu_memory_demand(batch_size * sample_size);
        demand += decltype(negative_batch)::gpu_memory_demand(batch_size * num_negative);
#ifdef USE_LOSS
        demand += decltype(loss)::gpu_memory_demand(batch_size);
#endif
        demand += decltype(random)::gpu_memory_demand(batch_size * num_negative * 2);
        demand += decltype(negative_sampler)::gpu_memory_demand(sampler_size);
        return demand;
    }
};

}; // namespace graphvite