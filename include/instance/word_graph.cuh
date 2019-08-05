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

#include <unordered_map>

#include "graph.cuh"

namespace graphvite {

/**
 * @brief Normal graphs of word co-occurrences
 *
 * Reference:
 *
 * 1) word2vec
 * https://papers.nips.cc/paper/5021-distributed-representations-of-words-
 * and-phrases-and-their-compositionality.pdf
 *
 * 2) LINE
 * https://arxiv.org/pdf/1503.03578.pdf
 *
 * @tparam _Index integral type of node indexes
 */
template<class _Index = size_t>
class WordGraph : public Graph<_Index> {
public:
    typedef Graph <_Index> Base;
    USING_GRAPH(Base);

    typedef _Index Index;

    /** Clear the graph and free CPU memory */
    void clear() {
        Base::clear();
        as_undirected = true;
    }

    inline std::string name() const override {
        std::stringstream ss;
        ss << "WordGraph<" << pretty::type2name<Index>() << ">";
        return ss.str();
    }

    /**
     * @brief Load a word graph from a corpus file. Store the graph in an adjacency list.
     *
     * Multiple occrruences of the same edge are treated as weight.
     *
     * @param file_name file name
     * @param window word pairs with distance <= window are counted as edges
     * @param min_count words with occurrence < min_count are dicarded
     * @param _normalization normalize the adjacency matrix or not
     * @param delimiters string of delimiter characters
     * @param comment prefix of comment strings
     */
    void load_file_compact(const char *file_name, int window = 5, int min_count = 5, bool _normalization = false,
                           const char *delimiters = " \t\r\n", const char *comment = "#") {
        LOG(INFO) << "generating graph from corpus " << file_name;
        clear();
        normalization = _normalization;

        FILE *fin = fopen(file_name, "r");
        CHECK(fin) << "File `" << file_name << "` doesn't exist";
        fseek(fin, 0, SEEK_END);
        size_t fsize = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        char line[kMaxLineLength];
        std::vector <Index> frequency;

        while (fgets(line, kMaxLineLength, fin) != nullptr) {
            LOG_EVERY_N(INFO, 1e7) << 50.0 * ftell(fin) / fsize << "%";

            char *comment_str = strstr(line, comment);
            if (comment_str)
                *comment_str = 0;

            for (char *word = strtok(line, delimiters); word; word = strtok(nullptr, delimiters)) {
                auto iter = name2id.find(word);
                if (iter != name2id.end())
                    frequency[iter->second]++;
                else {
                    name2id[word] = num_vertex++;
                    id2name.push_back(word);
                    frequency.push_back(1);
                }
            }
        }
        std::vector <std::string> id2name_new;
        name2id.clear();
        num_vertex = 0;
        for (Index i = 0; i < id2name.size(); i++)
            if (frequency[i] >= min_count) {
                auto &word = id2name[i];
                id2name_new.push_back(word);
                name2id[word] = num_vertex++;
            }
        id2name = id2name_new;
        vertex_edges.resize(num_vertex);
        vertex_weights.resize(num_vertex);

        fseek(fin, 0, SEEK_SET);
        std::vector <std::unordered_map<Index, float>> edge_map(num_vertex);
        std::vector <Index> sentence;

        while (fgets(line, kMaxLineLength, fin) != nullptr) {
            LOG_EVERY_N(INFO, 1e7) << 50 + 50.0 * ftell(fin) / fsize << "%";

            char *comment_str = strstr(line, comment);
            if (comment_str)
                *comment_str = 0;

            sentence.clear();
            for (char *word = strtok(line, delimiters); word; word = strtok(nullptr, delimiters)) {
                auto iter = name2id.find(word);
                if (iter != name2id.end())
                    sentence.push_back(iter->second);
            }
            for (int i = 0; i < sentence.size(); i++)
                for (int j = 1; j <= window; j++) {
                    if (i + j >= sentence.size())
                        break;
                    Index u = sentence[i];
                    Index v = sentence[i + j];
                    auto edge_iter = edge_map[u].find(v);
                    if (edge_iter == edge_map[u].end())
                        edge_map[u][v] = 1;
                    else
                        edge_iter->second++;
                    edge_iter = edge_map[v].find(u);
                    if (edge_iter == edge_map[v].end())
                        edge_map[v][u] = 1;
                    else
                        edge_iter->second++;
                    vertex_weights[u]++;
                    vertex_weights[v]++;
                }
        }
        fclose(fin);
        for (Index i = 0; i < num_vertex; i++) {
            for (auto &&edge : edge_map[i])
                vertex_edges[i].push_back(edge);
            num_edge += edge_map[i].size();
        }
        if (normalization)
            normalize();

        LOG(WARNING) << pretty::block(info());
    }

    /**
     * @brief Load a word graph from a corpus file. Store the graph in an adjacency list.
     * @deprecated This function consumes 5~10x memory than load_compact().
     *
     * Multiple occrruences of the same edge are treated as multiple edges.
     *
     * @param file_name file name
     * @param window words with distance <= window are counted as edges
     * @param min_count words with occurrence < min_count are dicarded
     * @param _normalization normalize the adjacency matrix or not
     * @param delimiters string of delimiter characters
     * @param comment prefix of comment strings
     */
    void load_file(const char *file_name, int window = 5, int min_count = 5, bool _normalization = false,
                   const char *delimiters = " \t\r\n", const char *comment = "#")

    DEPRECATED("This function consumes 5~10x memory than load_compact()") {
        LOG(INFO) << "generating graph from corpus " << file_name;
        clear();
        normalization = _normalization;

        FILE *fin = fopen(file_name, "r");
        CHECK(fin) << "File `" << file_name << "` doesn't exist";
        fseek(fin, 0, SEEK_END);
        size_t fsize = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        char line[kMaxLineLength];
        std::vector <Index> frequency;

        while (fgets(line, kMaxLineLength, fin) != nullptr) {
            LOG_EVERY_N(INFO, 1e7) << 50.0 * ftell(fin) / fsize << "%";

            char *comment_str = strstr(line, comment);
            if (comment_str)
                *comment_str = 0;

            for (char *word = strtok(line, delimiters); word; word = strtok(nullptr, delimiters)) {
                auto iter = name2id.find(word);
                if (iter != name2id.end())
                    frequency[iter->second]++;
                else {
                    name2id[word] = num_vertex++;
                    id2name.push_back(word);
                    frequency.push_back(1);
                }
            }
        }
        std::vector <std::string> id2name_new;
        name2id.clear();
        num_vertex = 0;
        for (Index i = 0; i < id2name.size(); i++)
            if (frequency[i] >= min_count) {
                auto &word = id2name[i];
                id2name_new.push_back(word);
                name2id[word] = num_vertex++;
            }
        id2name = id2name_new;
        vertex_edges.resize(num_vertex);
        vertex_weights.resize(num_vertex);

        fseek(fin, 0, SEEK_SET);
        std::vector <Index> sentence;

        while (fgets(line, kMaxLineLength, fin) != nullptr) {
            LOG_EVERY_N(INFO, 1e7) << 50 + 50.0 * ftell(fin) / fsize << "%";

            char *comment_str = strstr(line, comment);
            if (comment_str)
                *comment_str = 0;

            sentence.clear();
            for (char *word = strtok(line, delimiters); word; word = strtok(nullptr, delimiters)) {
                auto iter = name2id.find(word);
                if (iter != name2id.end())
                    sentence.push_back(iter->second);
            }
            for (int i = 0; i < sentence.size(); i++)
                for (int j = 1; j <= window; j++) {
                    if (i + j >= sentence.size())
                        break;
                    Index u = sentence[i];
                    Index v = sentence[i + j];
                    vertex_edges[u].push_back(std::make_tuple(v, 1));
                    vertex_edges[v].push_back(std::make_tuple(u, 1));
                    vertex_weights[u]++;
                    vertex_weights[v]++;
                    num_edge++;
                }
        }
        fclose(fin);
        if (normalization)
            normalize();

        LOG(WARNING) << pretty::block(info());
    }
};

} // namespace graphvite