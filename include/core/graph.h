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

#define FILE_OFFSET_BITS 64
#include <cstdio>
#undef FILE_OFFSET_BITS

#include <tuple>
#include <string>
#include <vector>
#include <glog/logging.h>

#include "util/common.h"
#include "util/debug.h"

namespace graphvite {

/**
 * @brief General interface of graphs
 * @tparam _Index integral type of node indexes
 * @tparam _Attributes types of additional edge attributes
 *
 * @note To add a new graph, you need to
 * - derive a template graph class from GraphMixin
 * - implement all virtual functions for that class
 * - add python binding of instantiations of that class in extension.h & extension.cu
 */
template<class _Index, class ..._Attributes>
class GraphMixin {
public:
    typedef _Index Index;
    typedef std::tuple<Index, float, _Attributes...> VertexEdge;
    typedef std::tuple<Index, Index, float, _Attributes...> Edge;

    std::vector<std::vector<VertexEdge>> vertex_edges;
    std::vector<Edge> edges;
    std::vector<float> vertex_weights, edge_weights;
    std::vector<size_t> flat_offsets;

    Index num_vertex;
    size_t num_edge;

#define USING_GRAPH_MIXIN(type) \
    using typename type::VertexEdge; \
    using typename type::Edge; \
    using type::vertex_edges; \
    using type::edges; \
    using type::vertex_weights; \
    using type::edge_weights; \
    using type::num_vertex; \
    using type::num_edge; \
    using type::info

    GraphMixin() = default;
    GraphMixin(const GraphMixin &) = delete;
    GraphMixin &operator=(const GraphMixin &) = delete;

    /** Clear the graph and free CPU memory */
    virtual void clear() {
        num_vertex = 0;
        num_edge = 0;
        decltype(vertex_edges)().swap(vertex_edges);
        decltype(edges)().swap(edges);
        decltype(vertex_weights)().swap(vertex_weights);
        decltype(edge_weights)().swap(edge_weights);
        decltype(flat_offsets)().swap(flat_offsets);
    }

    /** Flatten the adjacency list to an edge list */
    virtual void flatten() {
        if (!edges.empty())
            return;

        size_t offset = 0;
        for (Index u = 0; u < num_vertex; u++) {
            for (auto &&vertex_edge : vertex_edges[u]) {
                edges.push_back(std::tuple_cat(std::tie(u), vertex_edge));
                edge_weights.push_back(std::get<1>(vertex_edge));
            }
            flat_offsets.push_back(offset);
            offset += vertex_edges[u].size();
        }
    }

    virtual inline std::string name() const {
        std::stringstream ss;
        ss << "GraphMixin<" << pretty::type2name<Index>();
        auto _ = {0, (ss << ", " << pretty::type2name<_Attributes>(), 0)...};
        ss << ">";
        return ss.str();
    }

    virtual inline std::string graph_info() const {
        std::stringstream ss;
        ss << "#vertex: " << num_vertex << ", #edge: " << num_edge;
        return ss.str();
    }

    /** Return information about the graph */
    std::string info() {
        std::stringstream ss;
        ss << name() << std::endl;
        ss << pretty::header("Graph") << std::endl;
        ss << graph_info();
        return ss.str();
    }
};
} // namespace graphvite