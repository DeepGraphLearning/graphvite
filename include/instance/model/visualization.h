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

#include "core/optimizer.h"
#include "util/gpu.cuh"

namespace graphvite {

/**
 * @brief LargeVis model
 * @tparam _Vector vector type of embeddings
 *
 * Forward: L2_norm(head - tail) ^ 2
 * Backward: gradient of forward function
 */
template<class _Vector>
class LargeVis {
public:
    static const size_t dim = _Vector::dim;
    typedef _Vector Vector;
    typedef typename _Vector::Float Float;

    __host__ __device__
    static void forward(const Vector &head, const Vector &tail, Float &output) {
        output = 0;
        FOR(i, dim)
            output += (head[i] - tail[i]) * (head[i] - tail[i]);
        output = SUM(output);
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Float gradient, const Optimizer &optimizer, Float weight = 1) {
        auto update = get_update_function<Float, optimizer_type>();
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            head[i] -= (optimizer.*update)(h, gradient * (h - t), weight);
            tail[i] -= (optimizer.*update)(t, gradient * (t - h), weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &head_moment1, Vector &tail_moment1,
                         Float gradient, const Optimizer &optimizer, Float weight = 1) {
        auto update = get_update_function_1_moment<Float, optimizer_type>();
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            head[i] -= (optimizer.*update)(h, gradient * (h - t), head_moment1[i], weight);
            tail[i] -= (optimizer.*update)(t, gradient * (t - h), tail_moment1[i], weight);
        }
    }

    template<OptimizerType optimizer_type>
    __host__ __device__
    static void backward(Vector &head, Vector &tail, Vector &head_moment1, Vector &tail_moment1,
                         Vector &head_moment2, Vector &tail_moment2,
                         Float gradient, const Optimizer &optimizer, Float weight = 1) {
        auto update = get_update_function_2_moment<Float, optimizer_type>();
        FOR(i, dim) {
            Float h = head[i];
            Float t = tail[i];
            head[i] -= (optimizer.*update)(h, gradient * (h - t), head_moment1[i], head_moment2[i], weight);
            tail[i] -= (optimizer.*update)(t, gradient * (t - h), tail_moment1[i], tail_moment2[i], weight);
        }
    }
};

}