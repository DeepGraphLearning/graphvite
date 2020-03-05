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

#include "util/common.h"

//#define USE_TIMER
//#define PINNED_MEMORY

#include "bind.h"

const std::string version = "0.2.2";

PYBIND11_MODULE(libgraphvite, module) {
    py::options options;
    options.disable_function_signatures();

    // optimizers
    auto optimizer = module.def_submodule("optimizer");
    pyLRSchedule(optimizer, "LRSchedule");
    pyOptimizer(optimizer, "Optimizer");
    pySGD(optimizer, "SGD");
    pyMomentum(optimizer, "Momentum");
    pyAdaGrad(optimizer, "AdaGrad");
    pyRMSprop(optimizer, "RMSprop");
    pyAdam(optimizer, "Adam");

    // graphs
    auto graph = module.def_submodule("graph");
    pyGraph<unsigned int>(graph, "Graph");
    pyWordGraph<unsigned int>(graph, "WordGraph");
    pyKnowledgeGraph<unsigned int>(graph, "KnowledgeGraph");
    pyKNNGraph<unsigned int>(graph, "KNNGraph");

    // solvers
    auto solver = module.def_submodule("solver");

    pyGraphSolver<128, float, unsigned int>(solver, "GraphSolver");
#ifndef FAST_COMPILE
    pyGraphSolver<32, float, unsigned int>(solver, "GraphSolver");
    pyGraphSolver<64, float, unsigned int>(solver, "GraphSolver");
    pyGraphSolver<96, float, unsigned int>(solver, "GraphSolver");
    pyGraphSolver<256, float, unsigned int>(solver, "GraphSolver");
    pyGraphSolver<512, float, unsigned int>(solver, "GraphSolver");
#endif

    pyKnowledgeGraphSolver<512, float, unsigned int>(solver, "KnowledgeGraphSolver");
    pyKnowledgeGraphSolver<1024, float, unsigned int>(solver, "KnowledgeGraphSolver");
    pyKnowledgeGraphSolver<2048, float, unsigned int>(solver, "KnowledgeGraphSolver");
#ifndef FAST_COMPILE
    pyKnowledgeGraphSolver<32, float, unsigned int>(solver, "KnowledgeGraphSolver");
    pyKnowledgeGraphSolver<64, float, unsigned int>(solver, "KnowledgeGraphSolver");
    pyKnowledgeGraphSolver<96, float, unsigned int>(solver, "KnowledgeGraphSolver");
    pyKnowledgeGraphSolver<128, float, unsigned int>(solver, "KnowledgeGraphSolver");
    pyKnowledgeGraphSolver<256, float, unsigned int>(solver, "KnowledgeGraphSolver");
#endif

    pyVisualizationSolver<2, float, unsigned int>(solver, "VisualizationSolver");
#ifndef FAST_COMPILE
    pyVisualizationSolver<3, float, unsigned int>(solver, "VisualizationSolver");
#endif

    // interface
    py::enum_<DType> pyDType(module, "dtype");
    pyDType.value("uint32", DType::uint32)
           .value("uint64", DType::uint64)
           .value("float32", DType::float32)
           .value("float64", DType::float64);
    module.attr("dtype2name") = dtype2name;

    // glog
    module.def("init_logging", graphvite::init_logging, py::no_gil(),
               py::arg("threshhold") = google::INFO, py::arg("dir") = "", py::arg("verbose") = false);
    module.attr("INFO") = google::INFO;
    module.attr("WARNING") = google::WARNING;
    module.attr("ERROR") = google::ERROR;
    module.attr("FATAL") = google::FATAL;

    // io
    auto io = module.def_submodule("io");
    io.def("size_string", graphvite::pretty::size_string, py::no_gil(), py::arg("size"));
    io.def("yes_no", graphvite::pretty::yes_no, py::no_gil(), py::arg("x"));
    io.def("block", graphvite::pretty::block, py::no_gil(), py::arg("content"));
    io.def("header", graphvite::pretty::header, py::no_gil(), py::arg("content"));

    module.attr("auto") = graphvite::kAuto;
    module.def("KiB", graphvite::KiB, py::no_gil(), py::arg("size"));
    module.def("MiB", graphvite::MiB, py::no_gil(), py::arg("size"));
    module.def("GiB", graphvite::GiB, py::no_gil(), py::arg("size"));

    module.attr("__version__") = version;
}