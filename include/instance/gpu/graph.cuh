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
namespace graph {

/**
 * @brief Train node embedding with 0-moment optimizers
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 * @tparam optimizer_type type of optimizer
 */
template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
__global__ void train(Memory<Vector, Index> vertex_embeddings, Memory<Vector, Index> context_embeddings,
                      Memory<Index, int> batch, Memory<Index, int> negative_batch,
                      Memory<typename Vector::Float, int> loss,
                      Optimizer optimizer, float negative_weight) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int batch_size = batch.count / 2;
    const int num_negative = negative_batch.count / batch_size;
    Model<Vector> model;

    __shared__ Vector buffer[kThreadPerBlock / kWarpSize];
    Vector &vertex_buffer = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {tail, head}
        Index head_id = batch[sample_id * 2 + 1];
        Vector &vertex = vertex_embeddings[head_id];
        vertex_buffer = vertex;
        Float sample_loss = 0;

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
            Float logit;
            model.forward(vertex_buffer, context, logit);
            Float prob = sigmoid(logit);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
                sample_loss += weight * -log(prob + kEpsilon);
            } else {
                gradient = prob;
                weight = negative_weight;
                sample_loss += weight * -log(1 - prob + kEpsilon);
            }
            model.backward<optimizer_type>(vertex_buffer, context, gradient, optimizer, weight);
        }

        if (lane_id == 0)
            loss[sample_id] = sample_loss / (1 + num_negative * negative_weight);
        vertex = vertex_buffer;
    }
}

/**
 * @brief Train node embedding with 1-moment optimizers
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 * @tparam optimizer_type type of optimizer
 */
template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
__global__ void train_1_moment(Memory <Vector, Index> vertex_embeddings, Memory <Vector, Index> context_embeddings,
                               Memory<Vector, Index> vertex_moment1s, Memory<Vector, Index> context_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Memory<typename Vector::Float, int> loss,
                               Optimizer optimizer, float negative_weight) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int batch_size = batch.count / 2;
    const int num_negative = negative_batch.count / batch_size;
    Model<Vector> model;

    __shared__ Vector buffer[kThreadPerBlock / kWarpSize];
    Vector &vertex_buffer = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {tail, head}
        Index head_id = batch[sample_id * 2 + 1];
        Vector &vertex = vertex_embeddings[head_id];
        Vector &vertex_moment1 = vertex_moment1s[head_id];
        vertex_buffer = vertex;
        Float sample_loss = 0;

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
            Float logit;
            model.forward(vertex_buffer, context, logit);
            Float prob = sigmoid(logit);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
                sample_loss += weight * -log(prob + kEpsilon);
            } else {
                gradient = prob;
                weight = negative_weight;
                sample_loss += weight * -log(1 - prob + kEpsilon);
            }
            model.backward<optimizer_type>(vertex_buffer, context, vertex_moment1, context_moment1,
                                           gradient, optimizer, weight);
        }

        if (lane_id == 0)
            loss[sample_id] = sample_loss / (1 + num_negative * negative_weight);
        vertex = vertex_buffer;
    }
}

/**
 * @brief Train node embedding with 2-moment optimizers
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 * @tparam optimizer_type type of optimizer
 */
template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
__global__ void train_2_moment(Memory<Vector, Index> vertex_embeddings, Memory <Vector, Index> context_embeddings,
                               Memory<Vector, Index> vertex_moment1s, Memory<Vector, Index> context_moment1s,
                               Memory<Vector, Index> vertex_moment2s, Memory<Vector, Index> context_moment2s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Memory<typename Vector::Float, int> loss,
                               Optimizer optimizer, float negative_weight) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int batch_size = batch.count / 2;
    const int num_negative = negative_batch.count / batch_size;
    Model<Vector> model;

    __shared__ Vector buffer[kThreadPerBlock / kWarpSize];
    Vector &vertex_buffer = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {tail, head}
        Index head_id = batch[sample_id * 2 + 1];
        Vector &vertex = vertex_embeddings[head_id];
        Vector &vertex_moment1 = vertex_moment1s[head_id];
        Vector &vertex_moment2 = vertex_moment2s[head_id];
        vertex_buffer = vertex;
        Float sample_loss = 0;

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
            Float logit;
            model.forward(vertex_buffer, context, logit);
            Float prob = sigmoid(logit);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
                sample_loss += weight * -log(prob + kEpsilon);
            } else {
                gradient = prob;
                weight = negative_weight;
                sample_loss += weight * -log(1 - prob + kEpsilon);
            }
            model.backward<optimizer_type>(vertex_buffer, context, vertex_moment1, context_moment1,
                                           vertex_moment2, context_moment2, gradient, optimizer, weight);
        }

        if (lane_id == 0)
            loss[sample_id] = sample_loss / (1 + num_negative * negative_weight);
        vertex = vertex_buffer;
    }
}

/**
 * @brief Predict logits for batch samples
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 */
template<class Vector, class Index, template<class> class Model>
__global__ void predict(Memory<Vector, Index> vertex_embeddings, Memory<Vector, Index> context_embeddings,
                        Memory<Index, int> batch, Memory<typename Vector::Float, int> logits) {
    static const size_t dim = Vector::dim;
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int batch_size = batch.count / 2;
    Model<Vector> model;

    __shared__ Vector buffer[kThreadPerBlock / kWarpSize];
    Vector &vertex_buffer = buffer[threadIdx.x / kWarpSize];

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {tail, head}
        Index head_id = batch[sample_id * 2 + 1];
        Index tail_id = batch[sample_id * 2];
        Vector &vertex = vertex_embeddings[head_id];
        Vector &context = context_embeddings[tail_id];

        Float logit;
        model.forward(vertex, context, logit);

        if (lane_id == 0)
            logits[sample_id] = logit;
    }
}

} // namespace graph
} // namespace gpu
} // namespace graphvite