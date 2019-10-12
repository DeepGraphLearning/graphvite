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

namespace graphvite {

// helper macros for CPU-GPU agnostic code
#if __CUDA_ARCH__

#define FOR(i, stop) \
    const int lane_id = threadIdx.x % gpu::kWarpSize; \
    for (int i = lane_id; i < (stop); i += gpu::kWarpSize)
#define SUM(x) gpu::WarpBroadcast(gpu::WarpReduce(x), 0)

#else

#define FOR(i, stop) \
    for (int i = 0; i < stop; i++)
#define SUM(x) (x)

#endif

namespace gpu {

const int kBlockPerGrid = 8192;
const int kThreadPerBlock = 512;
const int kWarpSize = 32;
const unsigned kFullMask = 0xFFFFFFFF;

template<class T>
__device__ T WarpReduce(T value) {
#pragma unroll
    for (int delta = 1; delta < kWarpSize; delta *= 2)
#if __CUDACC_VER_MAJOR__ >= 9
        value += __shfl_down_sync(kFullMask, value, delta);
#else
        value += __shfl_down(value, delta);
#endif
    return value;
}

template<class T>
__device__ T WarpBroadcast(T value, int lane_id) {
#if __CUDACC_VER_MAJOR__ >= 9
    return __shfl_sync(kFullMask, value, lane_id);
#else
    return __shfl(value, lane_id);
#endif
}

} // namespace gpu
} // namespace graphvite