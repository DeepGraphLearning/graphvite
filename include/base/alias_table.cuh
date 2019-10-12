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

#include <vector>
#include <memory>
#include <queue>
#include <cuda_runtime.h>

#include "memory.h"

namespace graphvite {

template <class Float = float, class Index = size_t>
class AliasTable;

namespace gpu {

template <class Float, class Index>
__global__ void Sample(AliasTable<Float, Index> sampler, Memory<double, int> rand, Memory<Index, int> result);

}

/**
 * @brief CPU / GPU implementation of the alias table algorithm
 *
 * Generate a sample from any discrete distribution in O(1) time.
 *
 * @tparam _Float floating type of probability
 * @tparam _Index integral type of indexes
 */
template <class _Float, class _Index>
class AliasTable {
public:
    typedef _Float Float;
    typedef _Index Index;

    static const int kThreadPerBlock = 512;

    int device_id;
    Index count;
    cudaStream_t stream;
    Memory<Float, Index> prob_table;
    Memory<Index, Index> alias_table;

    /** @brief Construct an alias table
     * @param _device_id GPU id, -1 for CPU
     * @param _stream CUDA stream
     */
    AliasTable(int _device_id, cudaStream_t _stream = 0) :
            device_id(_device_id), count(0), stream(_stream), prob_table(device_id, 0, stream),
            alias_table(device_id, 0, stream) {}

    /** Shallow copy constructor */
    AliasTable(const AliasTable &a) :
            device_id(a.device_id), count(a.count), stream(a.stream), prob_table(a.prob_table),
            alias_table(a.alias_table) {}

    AliasTable &operator=(const AliasTable &) = delete;

    /** Reallocate the memory space */
    void reallocate(Index capacity) {
        prob_table.reallocate(capacity);
        alias_table.reallocate(capacity);
    }

    /** Initialize the table with a distribution */
    void build(const std::vector<Float> &_prob_table) {
        count = _prob_table.size();
        CHECK(count > 0) << "Invalid sampling distribution";
        prob_table.resize(count);
        alias_table.resize(count);

        memcpy(prob_table.host_ptr, _prob_table.data(), count * sizeof(Float));
        // single precision may cause considerable trunctation error
        double norm = 0;
        for (int i = 0; i < count; i++)
            norm += prob_table[i];
        norm = norm / count;
        for (int i = 0; i < count; i++)
            prob_table[i] /= norm;

        std::queue<Index> large, little;
        for (int i = 0; i < count; i++) {
            if (prob_table[i] < 1)
                little.push(i);
            else
                large.push(i);
        }
        while (!little.empty() && !large.empty()) {
            Index i = little.front(), j = large.front();
            little.pop();
            large.pop();
            alias_table[i] = j;
            prob_table[j] = prob_table[i] + prob_table[j] - 1;
            if (prob_table[j] < 1)
                little.push(j);
            else
                large.push(j);
        }
        // suppress some trunction error
        while (!little.empty()) {
            Index i = little.front();
            little.pop();
            alias_table[i] = i;
        }
        while (!large.empty()) {
            Index i = large.front();
            large.pop();
            alias_table[i] = i;
        }
    }

    /** Copy the table to GPU */
    void to_device() {
        prob_table.to_device();
        alias_table.to_device();
    }

    /** Copy the table to GPU (asynchronous) */
    void to_device_async() {
        prob_table.to_device_async();
        alias_table.to_device_async();
    }

    /** Free GPU memory */
    void clear() {
        reallocate(0);
    }

    /** Generate a sample on CPU / GPU */
    __host__ __device__ inline Index sample(double rand1, double rand2) const {
        Index index = rand1 * count;
        Float prob = rand2;
        return prob < prob_table[index] ? index : alias_table[index];
    }

    /** Generate a batch of samples on GPU */
    void device_sample(const Memory<double, int> &rand, Memory<Index, int> *result) {
        int block_per_grid = (result->count + kThreadPerBlock - 1) / kThreadPerBlock;
        gpu::Sample<Float, Index><<<block_per_grid, kThreadPerBlock, 0, stream>>>(*this, rand, *result);
    }

    /**
     * @param count size of the distribution
     * @return GPU memory cost
     */
    static size_t gpu_memory_demand(int count) {
        size_t demand = 0;
        demand += decltype(prob_table)::gpu_memory_demand(count);
        demand += decltype(alias_table)::gpu_memory_demand(count);
        return demand;
    }
};

namespace gpu {

template <class Float, class Index>
__global__ void Sample(AliasTable<Float, Index> sampler, Memory<double, int> random, Memory<Index, int> result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < result.count) {
        Float rand1 = random[thread_id * 2];
        Float rand2 = random[thread_id * 2 + 1];
        result[thread_id] = sampler.sample(rand1, rand2);
    }
}

} // namespace gpu

} // namespace graphvite