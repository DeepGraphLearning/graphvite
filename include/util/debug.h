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

#include <curand.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace graphvite {

#define CUDA_CHECK(error) CudaCheck((error), __FILE__, __LINE__)
#define CURAND_CHECK(error) CurandCheck((error), __FILE__, __LINE__)

inline void CudaCheck(cudaError_t error, const char *file_name, int line) {
    CHECK(error == cudaSuccess)
            << "CUDA error " << cudaGetErrorString(error) << " at " << file_name << ":" << line;
}

inline void CurandCheck(curandStatus_t error, const char *file_name, int line) {
    CHECK(error == CURAND_STATUS_SUCCESS)
            << "CURAND error " << error << " at " << file_name << ":" << line;
}

} // namespace graphvite