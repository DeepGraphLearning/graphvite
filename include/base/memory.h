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

#include <utility>
#include <cuda_runtime.h>

#include "util/debug.h"

namespace graphvite {

/**
 * @brief CPU / GPU memory space allocator
 * @tparam _T type of data
 * @tparam _Index integral type of indexes
 */
template<class _T, class _Index = size_t>
class Memory {
public:
    typedef _T Data;
    typedef _Index Index;

    int device_id;
    Index count = 0, capacity = 0;
    cudaStream_t stream;
    int *refer_count = nullptr;
    Data *host_ptr = nullptr, *device_ptr = nullptr;

    /**
     * @brief Construct a memory space
     * @param _device_id GPU id, -1 for CPU
     * @param _count number of data
     * @param _stream CUDA stream
     */
    Memory(int _device_id, Index _count = 0, cudaStream_t _stream = 0) :
            device_id(_device_id), stream(_stream) {
        resize(_count);
    }

    /** Shallow copy constructor */
    Memory(const Memory &m) :
            device_id(m.device_id), count(m.count), capacity(m.capacity), stream(m.stream), refer_count(m.refer_count),
            host_ptr(m.host_ptr), device_ptr(m.device_ptr) {
        if (capacity)
            (*refer_count)++;
    }

    Memory &operator=(const Memory &) = delete;

    ~Memory() { reallocate(0); }

    /** Swap two memory spaces */
    void swap(Memory &m) {
        std::swap(device_id, m.device_id);
        std::swap(count, m.count);
        std::swap(capacity, m.capacity);
        std::swap(stream, m.stream);
        std::swap(refer_count, m.refer_count);
        std::swap(host_ptr, m.host_ptr);
        std::swap(device_ptr, m.device_ptr);
    }

    __host__ __device__ Data &operator[](Index index) {
#ifdef __CUDA_ARCH__
        return device_ptr[index];
#else
        return host_ptr[index];
#endif
    }

    __host__ __device__ Data &operator[](Index index) const {
#ifdef __CUDA_ARCH__
        return device_ptr[index];
#else
        return host_ptr[index];
#endif
    }

    /** Copy data from another memory */
    void copy(const Memory &m) {
        resize(m.count);
        memcpy(host_ptr, m.host_ptr, count * sizeof(Data));
    }

    /** Copy data from a pointer */
    void copy(void *ptr, Index _count) {
        resize(_count);
        memcpy(host_ptr, ptr, count * sizeof(Data));
    }

    /** Reallocate the memory space */
    void reallocate(Index _capacity) {
        if (capacity && !--(*refer_count)) {
            delete refer_count;
#ifdef PINNED_MEMORY
            CUDA_CHECK(cudaFreeHost(host_ptr));
#else
            delete [] host_ptr;
#endif
            if (device_id != -1) {
                CUDA_CHECK(cudaSetDevice(device_id));
                CUDA_CHECK(cudaFree(device_ptr));
            }
        }
        capacity = _capacity;
        if (capacity) {
            refer_count = new int(1);
#ifdef PINNED_MEMORY
            CUDA_CHECK(cudaMallocHost(&host_ptr, capacity * sizeof(Data)));
#else
            host_ptr = new Data[capacity];
#endif
            if (device_id != -1) {
                CUDA_CHECK(cudaSetDevice(device_id));
                CUDA_CHECK(cudaMalloc(&device_ptr, capacity * sizeof(Data)));
            }
        }
    }

    /** Resize the memory space. Reallocate only if the capacity is not enough. */
    void resize(Index _count) {
        if (_count > capacity || (capacity && *refer_count > 1))
            reallocate(_count);
        count = _count;
    }

    /** Copy the memory space to GPU */
    void to_device(Index copy_count = 0) {
        if (count && device_id != -1) {
            if (!copy_count)
                copy_count = count;
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, copy_count * sizeof(Data), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    /** Copy the memory space to GPU (asynchronous) */
    void to_device_async(Index copy_count = 0) {
        if (count && device_id != -1) {
            if (!copy_count)
                copy_count = count;
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, copy_count * sizeof(Data), cudaMemcpyHostToDevice, stream));
        }
    }

    /** Copy the memory space back from GPU */
    void to_host(Index copy_count = 0) {
        if (count && device_id != -1) {
            if (!copy_count)
                copy_count = count;
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, copy_count * sizeof(Data), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    /** Copy the memory space back from GPU (asynchronous) */
    void to_host_async(Index copy_count = 0) {
        if (count && device_id != -1) {
            if (!copy_count)
                copy_count = count;
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, copy_count * sizeof(Data), cudaMemcpyDeviceToHost, stream));
        }
    }

    /** Fill the memory space with data. Automatically resize the memory when necessary. */
    void fill(const Data &data, Index _count = 0) {
        if (_count)
            resize(_count);
        for (Index i = 0; i < count; i++)
            host_ptr[i] = data;
    }

    /** Gather data from a pool according to an index mapping. Automatically resize the memory when necessary. */
    void gather(const std::vector<Data> &memory, const std::vector<Index> &mapping) {
        if (!mapping.empty()) {
            resize(mapping.size());
            for (Index i = 0; i < count; i++)
                host_ptr[i] = memory[mapping[i]];
        }
        else {
            resize(memory.size());
            for (Index i = 0; i < count; i++)
                host_ptr[i] = memory[i];
        }
    }

    /** Scatter data to a pool according to an index mapping */
    void scatter(std::vector<Data> &memory, const std::vector<Index> &mapping) {
        if (!mapping.empty()) {
            for (Index i = 0; i < count; i++)
                memory[mapping[i]] = host_ptr[i];
        }
        else {
            for (Index i = 0; i < count; i++)
                memory[i] = host_ptr[i];
        }
    }

    /** Scatter data to a pool by addition, according to an index mapping */
    void scatter_add(std::vector<Data> &memory, const std::vector<Index> &mapping) {
        if (!mapping.empty()) {
            for (Index i = 0; i < count; i++)
                memory[mapping[i]] += host_ptr[i];
        }
        else {
            for (Index i = 0; i < count; i++)
                memory[i] += host_ptr[i];
        }
    }

    /** Scatter data to a pool by substraction, according to an index mapping */
    void scatter_sub(std::vector<Data> &memory, const std::vector<Index> &mapping) {
        if (!mapping.empty()) {
            for (Index i = 0; i < count; i++)
                memory[mapping[i]] -= host_ptr[i];
        }
        else {
            for (Index i = 0; i < count; i++)
                memory[i] -= host_ptr[i];
        }
    }

    /**
     * @param capacity number of data
     * @return GPU memory cost
     */
    static size_t gpu_memory_demand(int capacity) {
        return capacity * sizeof(Data);
    }
};

} // namespace graphvite