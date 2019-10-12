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

#include <algorithm>
#include <type_traits>

namespace graphvite {

#ifndef __CUDA_ARCH__
using std::abs; // the template version of abs()
#endif

template<class Float>
__host__ __device__ Float sigmoid(Float x) {
    return x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
}

template<class Float>
__host__ __device__ Float safe_exp(Float x);

template<>
__host__ __device__ float safe_exp(float x) {
    static const float kLogitClip = 80;
#if __CUDA_ARCH__
    return exp(min(max(x, -kLogitClip), kLogitClip));
#else
    return std::exp(std::min(std::max(x, -kLogitClip), kLogitClip));
#endif
}

template<>
__host__ __device__ double safe_exp(double x) {
    static const double kLogitClip = 700;
#if __CUDA_ARCH__
    return exp(min(max(x, -kLogitClip), kLogitClip));
#else
    return std::exp(std::min(std::max(x, -kLogitClip), kLogitClip));
#endif
}

template<class Integer>
__host__ __device__ Integer bit_floor(Integer x) {
    static_assert(std::is_integral<Integer>::value, "bit_floor() can only be invoked with integral types");
#pragma unroll
    for (int i = 1; i < sizeof(Integer) * 8; i *= 2)
        x |= x >> i;
    return (x + 1) >> 1;
}

template<class Integer>
__host__ __device__ Integer bit_ceil(Integer x) {
    static_assert(std::is_integral<Integer>::value, "bit_ceil() can only be invoked with integral types");
    x--;
#pragma unroll
    for (int i = 1; i < sizeof(Integer) * 8; i *= 2)
        x |= x >> i;
    return x + 1;
}

} // namespace graphvie