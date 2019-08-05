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

#include <string>
#include <chrono>
#include <unordered_map>
#include <glog/logging.h>

namespace graphvite {

#ifdef USE_TIMER
class Timer {
public:
	typedef std::chrono::system_clock::time_point time_point;
	typedef std::chrono::high_resolution_clock clock;

	static std::unordered_map<std::string, int> occurrence;

	const char *prompt;
	int log_frequency;
	time_point start;

	Timer(const char *_prompt, int _log_frequency = 1)
		: prompt(_prompt), log_frequency(_log_frequency), start(clock::now()) {
		if (occurrence.find(prompt) == occurrence.end())
			occurrence[prompt] = 0;
	}

	~Timer() {
		time_point end = clock::now();
		LOG_IF(INFO, ++occurrence[prompt] == 1) << prompt << ": " << (end - start).count() / 1.0e6 << " ms";
		occurrence[prompt] %= log_frequency;
	}
};

std::unordered_map<std::string, int> Timer::occurrence;
#else
class Timer {
public:
    template<class... Args>
    Timer(const Args &...args) {}
};
#endif

} // namespace graphvite