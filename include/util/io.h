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

#include <sstream>
#include <glog/logging.h>

namespace graphvite {

void init_logging(int threshold = google::INFO, std::string dir = "", bool verbose = false) {
    static bool initialized = false;

    FLAGS_minloglevel = threshold;
    if (dir == "")
        FLAGS_logtostderr = true;
    else
        FLAGS_log_dir = dir;
    FLAGS_log_prefix = verbose;
    if (!initialized) {
        google::InitGoogleLogging("graphvite");
        initialized = true;
    }
}

namespace pretty {

template <class T>
std::string type2name();

template <>
std::string type2name<float>() { return "float32"; }

template <>
std::string type2name<double>() { return "float64"; }

template <>
std::string type2name<int>() { return "int32"; }

template <>
std::string type2name<unsigned int>() { return "uint32"; }

template <>
std::string type2name<long long>() { return "int64"; }

template <>
std::string type2name<unsigned long long>() { return "uint64"; }

std::string yes_no(bool x) {
    return x ? "yes" : "no";
}

std::string size_string(size_t x) {
    std::stringstream ss;
    ss.precision(3);
    if (x >= 1 << 30)
        ss << x / float(1 << 30) << " GiB";
    else if (x >= 1 << 20)
        ss << x / float(1 << 20) << " MiB";
    else if (x >= 1 << 10)
        ss << x / float(1 << 10) << " KiB";
    else
        ss << x << " B";
    return ss.str();
}

const size_t kLineWidth = 44;
std::string begin(kLineWidth, '<');
std::string end(kLineWidth, '>');

inline std::string block(const std::string &content) {
    std::stringstream ss;
    ss << begin << std::endl;
    ss << content << std::endl;
    ss << end << std::endl;
    return ss.str();
}

inline std::string header(const std::string &content) {
    std::stringstream ss;
    size_t padding = kLineWidth - content.length() - 2;
    std::string line(padding / 2, '-');
    ss << line << " " << content << " " << line;
    if (padding % 2 == 1)
        ss << '-';
    return ss.str();
}

} // namespace pretty
} // namespace graphvite