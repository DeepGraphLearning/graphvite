# Copyright 2019 MilaGraph. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Zhaocheng Zhu

"""Neural network definitions for applications"""
from __future__ import absolute_import

import types
import numpy as np

import torch
from torch import nn


class NodeClassifier(nn.Module):
    """
    Node classification network for graphs
    """
    def __init__(self, embedding, num_class, normalization=False):
        super(NodeClassifier, self).__init__()
        if normalization:
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = torch.as_tensor(embedding)
        self.embeddings = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.linear = nn.Linear(embedding.size(1), num_class, bias=True)

    def forward(self, indexes):
        x = self.embeddings(indexes)
        x = self.linear(x)
        return x


class LinkPredictor(nn.Module):
    """
    Link prediction network for graphs / knowledge graphs
    """
    def __init__(self, score_function, *embeddings, **kwargs):
        super(LinkPredictor, self).__init__()
        if isinstance(score_function, types.FunctionType):
            self.score_function = score_function
        else:
            self.score_function = getattr(LinkPredictor, score_function)
        self.kwargs = kwargs
        self.embeddings = nn.ModuleList()
        for embedding in embeddings:
            embedding = torch.as_tensor(embedding)
            embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
            self.embeddings.append(embedding)

    def forward(self, *indexes):
        assert len(indexes) == len(self.embeddings)
        vectors = []
        for index, embedding in zip(indexes, self.embeddings):
            vectors.append(embedding(index))
        return self.score_function(*vectors, **self.kwargs)

    @staticmethod
    def LINE(heads, tails):
        x = heads * tails
        score = x.sum(dim=1)
        return score

    DeepWalk = LINE

    @staticmethod
    def TransE(heads, relations, tails, margin=12):
        x = heads + relations - tails
        score = margin - x.norm(p=1, dim=1)
        return score

    @staticmethod
    def RotatE(heads, relations, tails, margin=12):
        dim = heads.size(1) // 2

        head_re, head_im = heads.view(-1, dim, 2).permute(2, 0, 1)
        tail_re, tail_im = tails.view(-1, dim, 2).permute(2, 0, 1)
        relations = relations[:, :dim]
        relation_re, relation_im = torch.cos(relations), torch.sin(relations)

        x_re = head_re * relation_re - head_im * relation_im - tail_re
        x_im = head_re * relation_im + head_im * relation_re - tail_im
        x = torch.stack([x_re, x_im], dim=0)
        score = margin - x.norm(p=2, dim=0).sum(dim=1)
        return score

    @staticmethod
    def DistMult(heads, relations, tails):
        x = heads * relations * tails
        score = x.sum(dim=1)
        return score

    @staticmethod
    def ComplEx(heads, relations, tails):
        dim = heads.size(1) // 2

        head_re, head_im = heads.view(-1, dim, 2).permute(2, 0, 1)
        tail_re, tail_im = tails.view(-1, dim, 2).permute(2, 0, 1)
        relation_re, relation_im = relations.view(-1, dim, 2).permute(2, 0, 1)

        x_re = head_re * relation_re - head_im * relation_im
        x_im = head_re * relation_im + head_im * relation_re
        x = x_re * tail_re + x_im * tail_im
        score = x.sum(dim=1)
        return score

    @staticmethod
    def SimplE(heads, relations, tails):
        dim = heads.size(1) // 2

        tails = tails.view(-1, dim, 2).flip(2).view(-1, dim * 2)

        x = heads * relations * tails
        score = x.sum(dim=1)
        return score

    @staticmethod
    def QuatE(heads, relations, tails):
        dim = heads.size(1) // 4

        head_r, head_i, head_j, head_k = heads.view(-1, dim, 4).permute(2, 0, 1)
        tail_r, tail_i, tail_j, tail_k = tails.view(-1, dim, 4).permute(2, 0, 1)
        relation_r, relation_i, relation_j, relation_k = relations.view(-1, dim, 4).permute(2, 0, 1)

        relation_norm = relations.view(-1, dim, 4).norm(p=2, dim=2)
        x_r = head_r * relation_r - head_i * relation_i - head_j * relation_j - head_k * relation_k
        x_i = head_r * relation_i + head_i * relation_r + head_j * relation_k - head_k * relation_j
        x_j = head_r * relation_j - head_i * relation_k + head_j * relation_r + head_k * relation_i
        x_k = head_r * relation_k + head_i * relation_j - head_j * relation_i + head_k * relation_r
        x = (x_r * tail_r + x_i * tail_i + x_j * tail_j + x_k * tail_k) / (relation_norm + 1e-15)
        score = x.sum(dim=1)
        return score