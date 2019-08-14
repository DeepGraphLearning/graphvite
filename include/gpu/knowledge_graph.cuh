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

#include "base/memory.h"
#include "core/optimizer.h"
#include "util/gpu.cuh"

namespace graphvite {
namespace gpu {

namespace transe {

/**
 * @brief Train TransE with 0-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                      Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                      Memory<Index, int> batch, Memory<Index, int> negative_batch,
                      Optimizer optimizer, float margin, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function<Float, type>();

    __shared__ graphvite::Vector<dim, bool> buffer[kThreadPerBlock / kWarpSize];

    graphvite::Vector<dim, bool> &sign = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize)
                    x += abs(head[i] + relation[i] - tail[i]);
                x = WarpBroadcast(WarpReduce(x), 0);
                x = margin - x;
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &tail = tail_embeddings[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float y = head[i] + relation[i] - tail[i];
                sign[i] = y > 0;
                x += abs(y);
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            x = margin - x;
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float s = sign[i] ? 1 : -1;
                head[i] -= (optimizer.*update)(head[i], -gradient * s, weight);
                tail[i] -= (optimizer.*update)(tail[i], gradient * s, weight);
                Float relation_update = (optimizer.*update)(relation[i], -gradient * s, weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train TransE with 1-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_1_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float margin, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function_1_moment<Float, type>();

    __shared__ graphvite::Vector<dim, bool> buffer[kThreadPerBlock / kWarpSize];

    graphvite::Vector<dim, bool> &sign = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize)
                    x += abs(head[i] + relation[i] - tail[i]);
                x = WarpBroadcast(WarpReduce(x), 0);
                x = margin - x;
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float y = head[i] + relation[i] - tail[i];
                sign[i] = y > 0;
                x += abs(y);
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            x = margin - x;
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float s = sign[i] ? 1 : -1;
                head[i] -= (optimizer.*update)(head[i], -gradient * s, head_moment1[i], weight);
                tail[i] -= (optimizer.*update)(tail[i], gradient * s, tail_moment1[i], weight);
                Float relation_update = (optimizer.*update)(relation[i], -gradient * s,
                                                            relation_moment1[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train TransE with 2-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_2_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s, Memory<Vector, Index> head_moment2s,
                               Memory<Vector, Index> tail_moment2s, Memory<Vector, Index> relation_moment2s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float margin, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function_2_moment<Float, type>();

    __shared__ graphvite::Vector<dim, bool> buffer[kThreadPerBlock / kWarpSize];

    graphvite::Vector<dim, bool> &sign = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_moment2 = relation_moment2s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize)
                    x += abs(head[i] + relation[i] - tail[i]);
                x = WarpBroadcast(WarpReduce(x), 0);
                x = margin - x;
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &head_moment2 = head_moment2s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            Vector &tail_moment2 = tail_moment2s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float y = head[i] + relation[i] - tail[i];
                sign[i] = y > 0;
                x += abs(y);
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            x = margin - x;
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float s = sign[i] ? 1 : -1;
                head[i] -= (optimizer.*update)(head[i], -gradient * s, head_moment1[i], head_moment2[i], weight);
                tail[i] -= (optimizer.*update)(tail[i], gradient * s, tail_moment1[i], tail_moment2[i], weight);
                Float relation_update = (optimizer.*update)(relation[i], -gradient * s,
                                                            relation_moment1[i], relation_moment2[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}
} // namespace transe

namespace distmult {

/**
 * @brief Train DistMult with 0-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                      Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                      Memory<Index, int> batch, Memory<Index, int> negative_batch,
                      Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize)
                    x += head[i] * relation[i] * tail[i];
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &tail = tail_embeddings[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize)
                x += head[i] * relation[i] * tail[i];
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float h = head[i];
                Float r = relation[i];
                Float t = tail[i];
                head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h, weight);
                tail[i] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t, weight);
                Float relation_update = (optimizer.*update)
                        (r, gradient * h * t + l3_regularization * abs(r) * r, weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train DistMult with 1-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_1_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function_1_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize)
                    x += head[i] * relation[i] * tail[i];
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize)
                x += head[i] * relation[i] * tail[i];
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float h = head[i];
                Float r = relation[i];
                Float t = tail[i];
                head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                               head_moment1[i], weight);
                tail[i] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                               tail_moment1[i], weight);
                Float relation_update = (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                                            relation_moment1[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train DistMult with 2-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_2_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s, Memory<Vector, Index> head_moment2s,
                               Memory<Vector, Index> tail_moment2s, Memory<Vector, Index> relation_moment2s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function_2_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_moment2 = relation_moment2s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize)
                    x += head[i] * relation[i] * tail[i];
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &head_moment2 = head_moment2s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            Vector &tail_moment2 = tail_moment2s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize)
                x += head[i] * relation[i] * tail[i];
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float h = head[i];
                Float r = relation[i];
                Float t = tail[i];
                head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                               head_moment1[i], head_moment2[i], weight);
                tail[i] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                               tail_moment1[i], tail_moment2[i], weight);
                Float relation_update = (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                                            relation_moment1[i], relation_moment2[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}
} // namespace distmult

namespace complex {

/**
 * @brief Train ComplEx with 0-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                      Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                      Memory<Index, int> batch, Memory<Index, int> negative_batch,
                      Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim / 2;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Float head_re = head[i * 2];
                    Float head_im = head[i * 2 + 1];
                    Float tail_re = tail[i * 2];
                    Float tail_im = tail[i * 2 + 1];
                    Float relation_re = relation[i * 2];
                    Float relation_im = relation[i * 2 + 1];
                    Float product_re = head_re * relation_re - head_im * relation_im;
                    Float product_im = head_re * relation_im + head_im * relation_re;
                    x += product_re * tail_re + product_im * tail_im;
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &tail = tail_embeddings[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_re = relation[i * 2];
                Float relation_im = relation[i * 2 + 1];
                Float product_re = head_re * relation_re - head_im * relation_im;
                Float product_im = head_re * relation_im + head_im * relation_re;
                x += product_re * tail_re + product_im * tail_im;
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_re = relation[i * 2];
                Float relation_im = relation[i * 2 + 1];
                // head
                Float head_re_grad = gradient * (relation_re * tail_re + relation_im * tail_im);
                Float head_im_grad = gradient * (-relation_im * tail_re + relation_re * tail_im);
                head[i * 2] -= (optimizer.*update)
                        (head_re, head_re_grad + l3_regularization * abs(head_re) * head_re, weight);
                head[i * 2 + 1] -= (optimizer.*update)
                        (head_im, head_im_grad + l3_regularization * abs(head_im) * head_im, weight);
                // tail
                Float tail_re_grad = gradient * (head_re * relation_re - head_im * relation_im);
                Float tail_im_grad = gradient * (head_re * relation_im + head_im * relation_re);
                tail[i * 2] -= (optimizer.*update)
                        (tail_re, tail_re_grad + l3_regularization * abs(tail_re) * tail_re, weight);
                tail[i * 2 + 1] -= (optimizer.*update)
                        (tail_im, tail_im_grad + l3_regularization * abs(tail_im) * tail_im, weight);
                // relation
                Float relation_re_grad = gradient * (head_re * tail_re + head_im * tail_im);
                Float relation_im_grad = gradient * (-head_im * tail_re + head_re * tail_im);
                Float relation_re_update = (optimizer.*update)
                        (relation_re, relation_re_grad + l3_regularization * abs(relation_re) * relation_re, weight);
                Float relation_im_update = (optimizer.*update)
                        (relation_im, relation_im_grad + l3_regularization * abs(relation_im) * relation_im, weight);
                relation[i * 2] -= relation_re_update;
                relation[i * 2 + 1] -= relation_im_update;
                relation_gradient[i * 2] += relation_re_update;
                relation_gradient[i * 2 + 1] += relation_im_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train ComplEx with 1-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_1_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim / 2;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function_1_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Float head_re = head[i * 2];
                    Float head_im = head[i * 2 + 1];
                    Float tail_re = tail[i * 2];
                    Float tail_im = tail[i * 2 + 1];
                    Float relation_re = relation[i * 2];
                    Float relation_im = relation[i * 2 + 1];
                    Float product_re = head_re * relation_re - head_im * relation_im;
                    Float product_im = head_re * relation_im + head_im * relation_re;
                    x += product_re * tail_re + product_im * tail_im;
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_re = relation[i * 2];
                Float relation_im = relation[i * 2 + 1];
                Float product_re = head_re * relation_re - head_im * relation_im;
                Float product_im = head_re * relation_im + head_im * relation_re;
                x += product_re * tail_re + product_im * tail_im;
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_re = relation[i * 2];
                Float relation_im = relation[i * 2 + 1];
                // head
                Float head_re_grad = gradient * (relation_re * tail_re + relation_im * tail_im);
                Float head_im_grad = gradient * (-relation_im * tail_re + relation_re * tail_im);
                head[i * 2] -= (optimizer.*update)(head_re, head_re_grad + l3_regularization * abs(head_re) * head_re,
                                                   head_moment1[i * 2], weight);
                head[i * 2 + 1] -= (optimizer.*update)
                        (head_im, head_im_grad + l3_regularization * abs(head_im) * head_im,
                         head_moment1[i * 2 + 1], weight);
                // tail
                Float tail_re_grad = gradient * (head_re * relation_re - head_im * relation_im);
                Float tail_im_grad = gradient * (head_re * relation_im + head_im * relation_re);
                tail[i * 2] -= (optimizer.*update)(tail_re, tail_re_grad + l3_regularization * abs(tail_re) * tail_re,
                                                   tail_moment1[i * 2], weight);
                tail[i * 2 + 1] -= (optimizer.*update)
                        (tail_im, tail_im_grad + l3_regularization * abs(tail_im) * tail_im,
                         tail_moment1[i * 2 + 1], weight);
                // relation
                Float relation_re_grad = gradient * (head_re * tail_re + head_im * tail_im);
                Float relation_im_grad = gradient * (-head_im * tail_re + head_re * tail_im);
                Float relation_re_update = (optimizer.*update)
                        (relation_re, relation_re_grad + l3_regularization * abs(relation_re) * relation_re,
                         relation_moment1[i], weight);
                Float relation_im_update = (optimizer.*update)
                        (relation_im, relation_im_grad + l3_regularization * abs(relation_im) * relation_im,
                         relation_moment1[i], weight);
                relation[i * 2] -= relation_re_update;
                relation[i * 2 + 1] -= relation_im_update;
                relation_gradient[i * 2] += relation_re_update;
                relation_gradient[i * 2 + 1] += relation_im_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train ComplEx with 2-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_2_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s, Memory<Vector, Index> head_moment2s,
                               Memory<Vector, Index> tail_moment2s, Memory<Vector, Index> relation_moment2s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim / 2;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function_2_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_moment2 = relation_moment2s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Float head_re = head[i * 2];
                    Float head_im = head[i * 2 + 1];
                    Float tail_re = tail[i * 2];
                    Float tail_im = tail[i * 2 + 1];
                    Float relation_re = relation[i * 2];
                    Float relation_im = relation[i * 2 + 1];
                    Float product_re = head_re * relation_re - head_im * relation_im;
                    Float product_im = head_re * relation_im + head_im * relation_re;
                    x += product_re * tail_re + product_im * tail_im;
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &head_moment2 = head_moment2s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            Vector &tail_moment2 = tail_moment2s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_re = relation[i * 2];
                Float relation_im = relation[i * 2 + 1];
                Float product_re = head_re * relation_re - head_im * relation_im;
                Float product_im = head_re * relation_im + head_im * relation_re;
                x += product_re * tail_re + product_im * tail_im;
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_re = relation[i * 2];
                Float relation_im = relation[i * 2 + 1];
                // head
                Float head_re_grad = gradient * (relation_re * tail_re + relation_im * tail_im);
                Float head_im_grad = gradient * (-relation_im * tail_re + relation_re * tail_im);
                head[i * 2] -= (optimizer.*update)(head_re, head_re_grad + l3_regularization * abs(head_re) * head_re,
                                                   head_moment1[i * 2], head_moment2[i * 2], weight);
                head[i * 2 + 1] -= (optimizer.*update)
                        (head_im, head_im_grad + l3_regularization * abs(head_im) * head_im,
                         head_moment1[i * 2 + 1], head_moment2[i * 2 + 1], weight);
                // tail
                Float tail_re_grad = gradient * (head_re * relation_re - head_im * relation_im);
                Float tail_im_grad = gradient * (head_re * relation_im + head_im * relation_re);
                tail[i * 2] -= (optimizer.*update)(tail_re, tail_re_grad + l3_regularization * abs(tail_re) * tail_re,
                                                   tail_moment1[i * 2], tail_moment2[i * 2], weight);
                tail[i * 2 + 1] -= (optimizer.*update)
                        (tail_im, tail_im_grad + l3_regularization * abs(tail_im) * tail_im,
                         tail_moment1[i * 2 + 1], tail_moment2[i * 2 + 1], weight);
                // relation
                Float relation_re_grad = gradient * (head_re * tail_re + head_im * tail_im);
                Float relation_im_grad = gradient * (-head_im * tail_re + head_re * tail_im);
                Float relation_re_update = (optimizer.*update)
                        (relation_re, relation_re_grad + l3_regularization * abs(relation_re) * relation_re,
                         relation_moment1[i], relation_moment2[i], weight);
                Float relation_im_update = (optimizer.*update)
                        (relation_im, relation_im_grad + l3_regularization * abs(relation_im) * relation_im,
                         relation_moment1[i], relation_moment2[i], weight);
                relation[i * 2] -= relation_re_update;
                relation[i * 2 + 1] -= relation_im_update;
                relation_gradient[i * 2] += relation_re_update;
                relation_gradient[i * 2 + 1] += relation_im_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}
} // namespace complex

namespace simple {

/**
 * @brief Train SimplE with 0-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                      Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                      Memory<Index, int> batch, Memory<Index, int> negative_batch,
                      Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Index j = i ^ 1;
                    x += head[i] * relation[i] * tail[j];
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &tail = tail_embeddings[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Index j = i ^ 1;
                x += head[i] * relation[i] * tail[j];
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Index j = i ^ 1;
                Float h = head[i];
                Float r = relation[i];
                Float t = tail[j];
                head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h, weight);
                tail[j] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t, weight);
                Float relation_update = (optimizer.*update)
                        (r, gradient * h * t + l3_regularization * abs(r) * r, weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train SimplE with 1-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_1_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function_1_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Index j = i ^ 1;
                    x += head[i] * relation[i] * tail[j];
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Index j = i ^ 1;
                x += head[i] * relation[i] * tail[j];
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Index j = i ^ 1;
                Float h = head[i];
                Float r = relation[i];
                Float t = tail[j];
                head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                               head_moment1[i], weight);
                tail[j] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                               tail_moment1[j], weight);
                Float relation_update = (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                                            relation_moment1[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train SimplE with 2-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_2_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s, Memory<Vector, Index> head_moment2s,
                               Memory<Vector, Index> tail_moment2s, Memory<Vector, Index> relation_moment2s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float l3_regularization, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;
    l3_regularization *= 3;

    auto update = get_update_function_2_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_moment2 = relation_moment2s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Index j = i ^ 1;
                    x += head[i] * relation[i] * tail[j];
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &head_moment2 = head_moment2s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            Vector &tail_moment2 = tail_moment2s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Index j = i ^ 1;
                x += head[i] * relation[i] * tail[j];
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Index j = i ^ 1;
                Float h = head[i];
                Float r = relation[i];
                Float t = tail[j];
                head[i] -= (optimizer.*update)(h, gradient * r * t + l3_regularization * abs(h) * h,
                                               head_moment1[i], head_moment2[i], weight);
                tail[j] -= (optimizer.*update)(t, gradient * h * r + l3_regularization * abs(t) * t,
                                               tail_moment1[j], tail_moment2[j], weight);
                Float relation_update = (optimizer.*update)(r, gradient * h * t + l3_regularization * abs(r) * r,
                                                            relation_moment1[i], relation_moment2[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}
} // namespace simple

namespace rotate {

/**
 * @brief Train RotatE with 0-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                      Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                      Memory<Index, int> batch, Memory<Index, int> negative_batch,
                      Optimizer optimizer, float margin, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim / 2;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Float head_re = head[i * 2];
                    Float head_im = head[i * 2 + 1];
                    Float tail_re = tail[i * 2];
                    Float tail_im = tail[i * 2 + 1];
                    Float relation_phase = relation[i];
                    Float relation_re = cos(relation_phase);
                    Float relation_im = sin(relation_phase);
                    Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                    Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                    x += sqrt(distance_re * distance_re + distance_im * distance_im);
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                x = margin - x;
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &tail = tail_embeddings[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_phase = relation[i];
                Float relation_re = cos(relation_phase);
                Float relation_im = sin(relation_phase);
                Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                x += sqrt(distance_re * distance_re + distance_im * distance_im);
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            x = margin - x;
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float relation_phase = relation[i];
                Float relation_re = cos(relation_phase);
                Float relation_im = sin(relation_phase);
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                Float grad_this_dim = gradient /
                                      (sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
                // head
                Float head_re_grad = -grad_this_dim * (distance_re * relation_re + distance_im * relation_im);
                Float head_im_grad = -grad_this_dim * (-distance_re * relation_im + distance_im * relation_re);
                head[i * 2] -= (optimizer.*update)(head_re, head_re_grad, weight);
                head[i * 2 + 1] -= (optimizer.*update)(head_im, head_im_grad, weight);
                // tail
                tail[i * 2] -= (optimizer.*update)(tail_re, grad_this_dim * distance_re, weight);
                tail[i * 2 + 1] -= (optimizer.*update)(tail_im, grad_this_dim * distance_im, weight);
                // relation
                Float relation_grad = -grad_this_dim *
                                      (distance_re * (head_re * -relation_im + head_im * -relation_re) +
                                       distance_im * (head_re * relation_re + head_im * -relation_im));
                Float relation_update = (optimizer.*update)(relation_phase, relation_grad, weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train RotatE with 1-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_1_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float margin, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim / 2;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function_1_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Float head_re = head[i * 2];
                    Float head_im = head[i * 2 + 1];
                    Float tail_re = tail[i * 2];
                    Float tail_im = tail[i * 2 + 1];
                    Float relation_phase = relation[i];
                    Float relation_re = cos(relation_phase);
                    Float relation_im = sin(relation_phase);
                    Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                    Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                    x += sqrt(distance_re * distance_re + distance_im * distance_im);
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                x = margin - x;
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_phase = relation[i];
                Float relation_re = cos(relation_phase);
                Float relation_im = sin(relation_phase);
                Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                x += sqrt(distance_re * distance_re + distance_im * distance_im);
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            x = margin - x;
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float relation_phase = relation[i];
                Float relation_re = cos(relation_phase);
                Float relation_im = sin(relation_phase);
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                Float grad_this_dim = gradient /
                                      (sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
                // head
                Float head_re_grad = -grad_this_dim * (distance_re * relation_re + distance_im * relation_im);
                Float head_im_grad = -grad_this_dim * (-distance_re * relation_im + distance_im * relation_re);
                head[i * 2] -= (optimizer.*update)(head_re, head_re_grad,
                                                   head_moment1[i * 2], weight);
                head[i * 2 + 1] -= (optimizer.*update)(head_im, head_im_grad,
                                                       head_moment1[i * 2 + 1], weight);
                // tail
                tail[i * 2] -= (optimizer.*update)(tail_re, grad_this_dim * distance_re,
                                                   tail_moment1[i * 2], weight);
                tail[i * 2 + 1] -= (optimizer.*update)(tail_im, grad_this_dim * distance_im,
                                                       tail_moment1[i * 2 + 1], weight);
                // relation
                Float relation_grad = -grad_this_dim *
                                      (distance_re * (head_re * -relation_im + head_im * -relation_re) +
                                       distance_im * (head_re * relation_re + head_im * -relation_im));
                Float relation_update = (optimizer.*update)(relation_phase, relation_grad,
                                                            relation_moment1[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}

/**
 * @brief Train RotatE with 2-moment optimizers
 *
 * Update protocols of embeddings
 * - head: in place
 * - relation: store gradients
 * - tail: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_2_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> relation_gradients,
                               Memory<Vector, Index> head_moment1s, Memory<Vector, Index> tail_moment1s,
                               Memory<Vector, Index> relation_moment1s, Memory<Vector, Index> head_moment2s,
                               Memory<Vector, Index> tail_moment2s, Memory<Vector, Index> relation_moment2s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Optimizer optimizer, float margin, float adversarial_temperature
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim / 2;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int num_head = head_embeddings.count;
    int batch_size = batch.count / 3;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function_2_moment<Float, type>();

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_moment2 = relation_moment2s[relation_id];
        Vector &relation_gradient = relation_gradients[relation_id];

        // compute normalizer
        Float x0, normalizer = 0;
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++) {
                Index head_id = batch[sample_id * 3 + 2];
                Index tail_id = batch[sample_id * 3 + 1];
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                Vector &head = head_embeddings[head_id];
                Vector &tail = tail_embeddings[tail_id];
                // Forward
                Float x = 0;
                for (int i = lane_id; i < dim; i += kWarpSize) {
                    Float head_re = head[i * 2];
                    Float head_im = head[i * 2 + 1];
                    Float tail_re = tail[i * 2];
                    Float tail_im = tail[i * 2 + 1];
                    Float relation_phase = relation[i];
                    Float relation_re = cos(relation_phase);
                    Float relation_im = sin(relation_phase);
                    Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                    Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                    x += sqrt(distance_re * distance_re + distance_im * distance_im);
                }
                x = WarpBroadcast(WarpReduce(x), 0);
                x = margin - x;
                if (s == 0)
                    x0 = x;
                normalizer += exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip));
            }

#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index head_id = batch[sample_id * 3 + 2];
            Index tail_id = batch[sample_id * 3 + 1];
            int label = 1;
            if (s < num_negative) {
                Index negative_id = negative_batch[sample_id * num_negative + s];
                if (negative_id < num_head)
                    head_id = negative_id;
                else
                    tail_id = negative_id - num_head;
                label = 0;
            }
            Vector &head = head_embeddings[head_id];
            Vector &head_moment1 = head_moment1s[head_id];
            Vector &head_moment2 = head_moment2s[head_id];
            Vector &tail = tail_embeddings[tail_id];
            Vector &tail_moment1 = tail_moment1s[tail_id];
            Vector &tail_moment2 = tail_moment2s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float relation_phase = relation[i];
                Float relation_re = cos(relation_phase);
                Float relation_im = sin(relation_phase);
                Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                x += sqrt(distance_re * distance_re + distance_im * distance_im);
            }
            x = WarpBroadcast(WarpReduce(x), 0);
            x = margin - x;
            Float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
#ifdef USE_LOSS
                sample_loss += weight * -log(prob + kEpsilon);
#endif
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon)
                    weight = exp(min(max((x - x0) / adversarial_temperature, -kLogitClip), kLogitClip)) / normalizer;
                else
                    weight = 1.0 / num_negative;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float relation_phase = relation[i];
                Float relation_re = cos(relation_phase);
                Float relation_im = sin(relation_phase);
                Float head_re = head[i * 2];
                Float head_im = head[i * 2 + 1];
                Float tail_re = tail[i * 2];
                Float tail_im = tail[i * 2 + 1];
                Float distance_re = head_re * relation_re - head_im * relation_im - tail_re;
                Float distance_im = head_re * relation_im + head_im * relation_re - tail_im;
                Float grad_this_dim = gradient /
                                      (sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
                // head
                Float head_re_grad = -grad_this_dim * (distance_re * relation_re + distance_im * relation_im);
                Float head_im_grad = -grad_this_dim * (-distance_re * relation_im + distance_im * relation_re);
                head[i * 2] -= (optimizer.*update)(head_re, head_re_grad,
                                                   head_moment1[i * 2], head_moment2[i * 2], weight);
                head[i * 2 + 1] -= (optimizer.*update)(head_im, head_im_grad,
                                                       head_moment1[i * 2 + 1], head_moment2[i * 2 + 1], weight);
                // tail
                tail[i * 2] -= (optimizer.*update)(tail_re, grad_this_dim * distance_re,
                                                   tail_moment1[i * 2], tail_moment2[i * 2], weight);
                tail[i * 2 + 1] -= (optimizer.*update)(tail_im, grad_this_dim * distance_im,
                                                       tail_moment1[i * 2 + 1], tail_moment2[i * 2 + 1], weight);
                // relation
                Float relation_grad = -grad_this_dim *
                                      (distance_re * (head_re * -relation_im + head_im * -relation_re) +
                                       distance_im * (head_re * relation_re + head_im * -relation_im));
                Float relation_update = (optimizer.*update)(relation_phase, relation_grad,
                                                            relation_moment1[i], relation_moment2[i], weight);
                relation[i] -= relation_update;
                relation_gradient[i] += relation_update;
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
#endif
    }
}
} // namespace rotate

}
}