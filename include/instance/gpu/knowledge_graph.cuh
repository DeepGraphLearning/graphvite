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
#include "util/math.h"

namespace graphvite {
namespace gpu {
namespace knowledge_graph {

/**
 * @brief Train knowledge graph embedding with 0-moment optimizers
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 * @tparam optimizer_type type of optimizer
 */
template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
__global__ void train(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                      Memory<Vector, Index> relation_embeddings, Memory<Index, int> batch,
                      Memory<Index, int> negative_batch, Memory<typename Vector::Float, int> loss,
                      Optimizer optimizer, float relation_lr_multiplier, float margin_or_l3,
                      float adversarial_temperature) {
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int num_head = head_embeddings.count;
    const int batch_size = batch.count / 3;
    const int num_negative = negative_batch.count / batch_size;
    Model<Vector> model;

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];

        // compute normalizer
        Float bias, normalizer = 0;
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
                Float logit;
                model.forward(head, tail, relation, logit, margin_or_l3);
                if (s == 0)
                    bias = logit;
                normalizer += safe_exp((logit - bias) / adversarial_temperature);
            }

        Float sample_loss = 0;
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
            Float logit;
            model.forward(head, tail, relation, logit, margin_or_l3);
            Float prob = sigmoid(logit);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
                sample_loss += weight * -log(prob + kEpsilon);
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon) {
                    weight = safe_exp((logit - bias) / adversarial_temperature) / normalizer;
                    // the normalizer may be out of date in ASGD
                    // so we need to clip the weight
                    weight = min(weight, Float(1));
                }
                else
                    weight = 1.0 / num_negative;
                sample_loss += weight * -log(1 - prob + kEpsilon);
            }
            model.backward<optimizer_type>(head, tail, relation,
                                           margin_or_l3, gradient, optimizer, relation_lr_multiplier, weight);
        }

        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
    }
}

/**
 * @brief Train knowledge graph embedding with 1-moment optimizers
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 * @tparam optimizer_type type of optimizer
 */
template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
__global__ void train_1_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> head_moment1s,
                               Memory<Vector, Index> tail_moment1s, Memory<Vector, Index> relation_moment1s,
                               Memory<Index, int> batch, Memory<Index, int> negative_batch,
                               Memory<typename Vector::Float, int> loss,
                               Optimizer optimizer, float relation_lr_multiplier, float margin_or_l3,
                               float adversarial_temperature) {
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int num_head = head_embeddings.count;
    const int batch_size = batch.count / 3;
    const int num_negative = negative_batch.count / batch_size;
    Model<Vector> model;

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];

        // compute normalizer
        Float bias, normalizer = 0;
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
                Float logit;
                model.forward(head, tail, relation, logit, margin_or_l3);
                if (s == 0)
                    bias = logit;
                normalizer += safe_exp((logit - bias) / adversarial_temperature);
            }

        Float sample_loss = 0;
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
            Float logit;
            model.forward(head, tail, relation, logit, margin_or_l3);
            Float prob = sigmoid(logit);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
                sample_loss += weight * -log(prob + kEpsilon);
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon) {
                    weight = safe_exp((logit - bias) / adversarial_temperature) / normalizer;
                    // the normalizer may be out of date in ASGD
                    // so we need to clip the weight
                    weight = min(weight, Float(1));
                }
                else
                    weight = 1.0 / num_negative;
                sample_loss += weight * -log(1 - prob + kEpsilon);
            }
            model.backward<optimizer_type>(head, tail, relation, head_moment1, tail_moment1, relation_moment1,
                                           margin_or_l3, gradient, optimizer, relation_lr_multiplier, weight);
        }

        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
    }
}

/**
 * @brief Train knowledge graph embedding with 2-moment optimizers
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 * @tparam optimizer_type type of optimizer
 */
template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
__global__ void train_2_moment(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                               Memory<Vector, Index> relation_embeddings, Memory<Vector, Index> head_moment1s,
                               Memory<Vector, Index> tail_moment1s, Memory<Vector, Index> relation_moment1s,
                               Memory<Vector, Index> head_moment2s, Memory<Vector, Index> tail_moment2s,
                               Memory<Vector, Index> relation_moment2s, Memory<Index, int> batch,
                               Memory<Index, int> negative_batch, Memory<typename Vector::Float, int> loss,
                               Optimizer optimizer, float relation_lr_multiplier, float margin_or_l3,
                               float adversarial_temperature) {
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int num_head = head_embeddings.count;
    const int batch_size = batch.count / 3;
    const int num_negative = negative_batch.count / batch_size;
    Model<Vector> model;

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index relation_id = batch[sample_id * 3];
        Vector &relation = relation_embeddings[relation_id];
        Vector &relation_moment1 = relation_moment1s[relation_id];
        Vector &relation_moment2 = relation_moment2s[relation_id];

        // compute normalizer
        Float bias, normalizer = 0;
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
                Float logit;
                model.forward(head, tail, relation, logit, margin_or_l3);
                if (s == 0)
                    bias = logit;
                normalizer += safe_exp((logit - bias) / adversarial_temperature);
            }

        Float sample_loss = 0;
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
            Float logit;
            model.forward(head, tail, relation, logit, margin_or_l3);
            Float prob = sigmoid(logit);
            // Backward
            Float gradient, weight;
            if (label) {
                gradient = prob - 1;
                weight = 1;
                sample_loss += weight * -log(prob + kEpsilon);
            } else {
                gradient = prob;
                if (adversarial_temperature > kEpsilon) {
                    weight = safe_exp((logit - bias) / adversarial_temperature) / normalizer;
                    // the normalizer may be out of date in ASGD
                    // so we need to clip the weight
                    weight = min(weight, Float(1));
                }
                else
                    weight = 1.0 / num_negative;
                sample_loss += weight * -log(1 - prob + kEpsilon);
            }
            model.backward<optimizer_type>(head, tail, relation, head_moment1, tail_moment1, relation_moment1,
                                           head_moment2, tail_moment2, relation_moment2,
                                           margin_or_l3, gradient, optimizer, relation_lr_multiplier, weight);
        }

        if (lane_id == 0)
            loss[sample_id] = sample_loss / 2;
    }
}

/**
 * @brief Predict logits for batch samples
 * @tparam Vector vector type of embeddings
 * @tparam Index integral type of indexes
 * @tparam Model embedding model
 */
template<class Vector, class Index, template<class> class Model>
__global__ void predict(Memory<Vector, Index> head_embeddings, Memory<Vector, Index> tail_embeddings,
                        Memory<Vector, Index> relation_embeddings, Memory<Index, int> batch,
                        Memory<typename Vector::Float, int> logits, float margin_or_l3) {
    typedef typename Vector::Float Float;

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % kWarpSize;
    const int num_thread = gridDim.x * blockDim.x;
    const int batch_size = batch.count / 3;
    Model<Vector> model;

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        // elements in std::tuple are stored in reverse order
        // each positive sample is {relation, tail, head}
        Index head_id = batch[sample_id * 3 + 2];
        Index tail_id = batch[sample_id * 3 + 1];
        Index relation_id = batch[sample_id * 3];
        Vector &head = head_embeddings[head_id];
        Vector &tail = tail_embeddings[tail_id];
        Vector &relation = relation_embeddings[relation_id];

        Float logit;
        model.forward(head, tail, relation, logit, margin_or_l3);

        if (lane_id == 0)
            logits[sample_id] = logit;
    }
}

} // namespace knowledge graph
} // namespace gpu
} // namespace graphvite