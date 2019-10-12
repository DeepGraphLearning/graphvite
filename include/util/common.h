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

#include "io.h"
#include "math.h"

namespace graphvite {

#define DEPRECATED(reason) __attribute__ ((deprecated(reason)))

const float kEpsilon = 1e-15;
const int kAuto = 0;
const size_t kMaxLineLength = 1 << 22;

constexpr size_t KiB(size_t x) {
    return x << 10;
}

constexpr size_t MiB(size_t x) {
    return x << 20;
}

constexpr size_t GiB(size_t x) {
    return x << 30;
}

} // namespace graphvite