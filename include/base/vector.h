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

#include <type_traits>
#include "util/gpu.cuh"

namespace graphvite {

/**
 * @brief Vector computation
 * @tparam _dim dimension
 * @tparam _Float floating type of data
 */
template<size_t _dim, class _Float = float>
class Vector {
     static_assert(std::is_floating_point<_Float>::value, "Vector can only be instantiated with floating point types");
    // static_assert(_dim % gpu::kWarpSize == 0, "`dim` should be divided by 32");
public:
    static const size_t dim = _dim;
    typedef size_t Index;
    typedef _Float Float;
    Float data[dim];

    /** Default constructor */
    Vector() = default;

    /** Construct a vector of repeat scalar */
    Vector(Float f) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] = f;
    }

    __host__ __device__ Float &operator[](Index index) {
        return data[index];
    }

    __host__ __device__ Float operator[](Index index) const {
        return data[index];
    }

    __host__ __device__ Vector &operator=(const Vector &v) {
#if __CUDA_ARCH__
        using namespace gpu;
        const int lane_id = threadIdx.x % kWarpSize;
        for (Index i = lane_id; i < dim; i += kWarpSize)
#else
        for (Index i = 0; i < dim; i++)
#endif
            data[i] = v[i];
        return *this;
    }

    Vector &operator =(Float f) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] = f;
        return *this;
    }

    Vector &operator +=(const Vector &v) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] += v[i];
        return *this;
    }


    Vector &operator -=(const Vector &v) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] -= v[i];
        return *this;
    }

    Vector &operator *=(const Vector &v) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] *= v[i];
        return *this;
    }

    Vector &operator /=(const Vector &v) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] /= v[i];
        return *this;
    }

    Vector &operator +=(Float f) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] += f;
        return *this;
    }

    Vector &operator -=(Float f) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] -= f;
        return *this;
    }

    Vector &operator *=(Float f) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] *= f;
        return *this;
    }

    Vector &operator /=(Float f) {
#pragma unroll
        for (Index i = 0; i < dim; i++)
            data[i] /= f;
        return *this;
    }

    Vector operator +(const Vector &v) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] + v[i];
        return result;
    }

    Vector operator -(const Vector &v) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] - v[i];
        return result;
    }

    Vector operator *(const Vector &v) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] * v[i];
        return result;
    }

    Vector operator /(const Vector &v) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] / v[i];
        return result;
    }

    Vector operator +(Float f) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] + f;
        return result;
    }

    Vector operator -(Float f) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] - f;
        return result;
    }

    Vector operator *(Float f) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] * f;
        return result;
    }

    Vector operator /(Float f) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = (*this)[i] / f;
        return result;
    }

    friend Vector operator +(Float f, const Vector &v) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = v[i] + f;
        return result;
    }

    friend Vector operator -(Float f, const Vector &v) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = v[i] - f;
        return result;
    }

    friend Vector operator *(Float f, const Vector &v) {
        Vector result;
#pragma unroll
        for (Index i = 0; i < dim; i++)
            result[i] = v[i] * f;
        return result;
    }
};

} // namespace graphvite