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
 * @author Zhaocheng Zhu, Shizhen Xu
 */

#pragma once

#include "base/memory.h"
#include "core/optimizer.h"
#include "util/gpu.cuh"

namespace graphvite {
namespace gpu {

namespace line {

/**
 * @brief Train LINE with 0-moment optimizers
 *
 * Update protocols of embeddings
 * - vertex: in place
 * - context: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train(Memory <Vector, Index> vertex_embeddings, Memory <Vector, Index> context_embeddings,
                      Memory<Index, int> batch, Memory<Index, int> negative_batch, Optimizer optimizer,
                      float negative_weight
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int batch_size = batch.count / 2;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function<Float, type>();

    __shared__ Vector buffer[kThreadPerBlock / kWarpSize];
    Vector &vertex_buffer = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {tail, head}
        Index head_id = batch[sample_id * 2 + 1];
        Vector &vertex = vertex_embeddings[head_id];
        for (int i = lane_id; i < dim; i += kWarpSize)
            vertex_buffer[i] = vertex[i];
#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index tail_id;
            int label;
            if (s < num_negative) {
                tail_id = negative_batch[sample_id * num_negative + s];
                label = 0;
            } else {
                tail_id = batch[sample_id * 2];
                label = 1;
            }
            Vector &context = context_embeddings[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize)
                x += vertex_buffer[i] * context[i];
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
                weight = negative_weight;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float v = vertex_buffer[i];
                Float c = context[i];
                vertex_buffer[i] -= (optimizer.*update)(v, gradient * c, weight);
                context[i] -= (optimizer.*update)(c, gradient * v, weight);
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / (1 + num_negative * negative_weight);
#endif
        for (int i = lane_id; i < dim; i += kWarpSize)
            vertex[i] = vertex_buffer[i];
    }
}

/**
 * @brief Train LINE with 1-moment optimizers
 *
 * Update protocols of embeddings
 * - vertex: in place
 * - context: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_1_moment(Memory <Vector, Index> vertex_embeddings, Memory <Vector, Index> context_embeddings,
                               Memory<Vector, Index> vertex_moment1s, Memory<Vector, Index> context_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch, Optimizer optimizer,
                               float negative_weight
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int batch_size = batch.count / 2;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function_1_moment<Float, type>();

    __shared__ Vector buffer[kThreadPerBlock / kWarpSize];
    Vector &vertex_buffer = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {tail, head}
        Index head_id = batch[sample_id * 2 + 1];
        Vector &vertex = vertex_embeddings[head_id];
        Vector &vertex_moment1 = vertex_moment1s[head_id];
        for (int i = lane_id; i < dim; i += kWarpSize)
            vertex_buffer[i] = vertex[i];
#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index tail_id;
            int label;
            if (s < num_negative) {
                tail_id = negative_batch[sample_id * num_negative + s];
                label = 0;
            } else {
                tail_id = batch[sample_id * 2];
                label = 1;
            }
            Vector &context = context_embeddings[tail_id];
            Vector &context_moment1 = context_moment1s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize)
                x += vertex_buffer[i] * context[i];
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
                weight = negative_weight;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float v = vertex_buffer[i];
                Float c = context[i];
                vertex_buffer[i] -= (optimizer.*update)(v, gradient * c, vertex_moment1[i], weight);
                context[i] -= (optimizer.*update)(c, gradient * v, context_moment1[i], weight);
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / (1 + num_negative * negative_weight);
#endif
        for (int i = lane_id; i < dim; i += kWarpSize)
            vertex[i] = vertex_buffer[i];
    }
}

/**
 * @brief Train LINE with 2-moment optimizers
 *
 * Update protocols of embeddings
 * - vertex: in place
 * - context: in place
 *
 * @tparam Vector type of embedding vectors
 * @tparam Index integral type of indexes
 * @tparam type type of optimizer
 */
template<class Vector, class Index, OptimizerType type>
__global__ void train_2_moment(Memory <Vector, Index> vertex_embeddings, Memory <Vector, Index> context_embeddings,
                               Memory<Vector, Index> vertex_moment1s, Memory<Vector, Index> context_moment1s,
                               Memory<Vector, Index> vertex_moment2s, Memory<Vector, Index> context_moment2s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch, Optimizer optimizer,
                               float negative_weight
#ifdef USE_LOSS
        , Memory<typename Vector::Float, int> loss
#endif
) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = thread_id % kWarpSize;
    int num_thread = gridDim.x * blockDim.x;
    int batch_size = batch.count / 2;
    int num_negative = negative_batch.count / batch_size;

    auto update = get_update_function_2_moment<Float, type>();

    __shared__ Vector buffer[kThreadPerBlock / kWarpSize];
    Vector &vertex_buffer = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {tail, head}
        Index head_id = batch[sample_id * 2 + 1];
        Vector &vertex = vertex_embeddings[head_id];
        Vector &vertex_moment1 = vertex_moment1s[head_id];
        Vector &vertex_moment2 = vertex_moment2s[head_id];
        for (int i = lane_id; i < dim; i += kWarpSize)
            vertex_buffer[i] = vertex[i];
#ifdef USE_LOSS
        Float sample_loss = 0;
#endif
        for (int s = 0; s <= num_negative; s++) {
            Index tail_id;
            int label;
            if (s < num_negative) {
                tail_id = negative_batch[sample_id * num_negative + s];
                label = 0;
            } else {
                tail_id = batch[sample_id * 2];
                label = 1;
            }
            Vector &context = context_embeddings[tail_id];
            Vector &context_moment1 = context_moment1s[tail_id];
            Vector &context_moment2 = context_moment2s[tail_id];
            // Forward
            Float x = 0;
            for (int i = lane_id; i < dim; i += kWarpSize)
                x += vertex_buffer[i] * context[i];
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
                weight = negative_weight;
#ifdef USE_LOSS
                sample_loss += weight * -log(1 - prob + kEpsilon);
#endif
            }
            for (int i = lane_id; i < dim; i += kWarpSize) {
                Float v = vertex_buffer[i];
                Float c = context[i];
                vertex_buffer[i] -= (optimizer.*update)(v, gradient * c, vertex_moment1[i], vertex_moment2[i], weight);
                context[i] -= (optimizer.*update)(c, gradient * v, context_moment1[i], context_moment2[i], weight);
            }
        }
#ifdef USE_LOSS
        if (lane_id == 0)
            loss[sample_id] = sample_loss / (1 + num_negative * negative_weight);
#endif
        for (int i = lane_id; i < dim; i += kWarpSize)
            vertex[i] = vertex_buffer[i];
    }
}
} // namespace line

namespace deepwalk = line;
namespace node2vec = line;

} // namespace gpu
} // namespace graphvite