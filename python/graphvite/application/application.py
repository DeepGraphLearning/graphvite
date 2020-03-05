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

"""Implementation of applications"""
from __future__ import print_function, absolute_import, unicode_literals, division

import os
import re
import pickle
import logging
import multiprocessing
from collections import defaultdict

from future.builtins import str, map, range
from easydict import EasyDict
import numpy as np

from .. import lib, cfg, auto
from .. import graph, solver
from ..util import assert_in, monitor, SharedNDArray

logger = logging.getLogger(__name__)


class ApplicationMixin(object):
    """
    General interface of graph applications.

    Parameters:
        dim (int): dimension of embeddings
        gpus (list of int, optional): GPU ids, default is all GPUs
        cpu_per_gpu (int, optional): number of CPU threads per GPU, default is all CPUs
        gpu_memory_limit (int, optional): memory limit per GPU in bytes, default is all memory
        float_type (dtype, optional): type of parameters
        index_type (dtype, optional): type of graph indexes
    """
    def __init__(self, dim, gpus=[], cpu_per_gpu=auto, gpu_memory_limit=auto,
                 float_type=cfg.float_type, index_type=cfg.index_type):
        self.dim = dim
        self.gpus = gpus
        self.cpu_per_gpu = cpu_per_gpu
        self.gpu_memory_limit = gpu_memory_limit
        self.float_type = float_type
        self.index_type = index_type
        self.set_format()

    def get_graph(self, **kwargs):
        raise NotImplementedError

    def get_solver(self, **kwargs):
        raise NotImplementedError

    def set_format(self, delimiters=" \t\r\n", comment="#"):
        """
        Set the format for parsing input data.

        Parameters:
            delimiters (str, optional): string of delimiter characters
            comment (str, optional): prefix of comment strings
        """
        self.delimiters = delimiters
        self.comment = comment
        self.pattern = re.compile("[%s]" % self.delimiters)

    @monitor.time
    def load(self, **kwargs):
        """load(**kwargs)
        Load a graph from file or Python object.
        Arguments depend on the underlying graph type.
        """
        self.graph = self.get_graph(**kwargs)
        if "file_name" in kwargs or "vector_file" in "kwargs":
            self.graph.load(delimiters=self.delimiters, comment=self.comment, **kwargs)
        else:
            self.graph.load(**kwargs)

    @monitor.time
    def build(self, **kwargs):
        """build(**kwargs)
        Build the solver from the graph.
        Arguments depend on the underlying solver type.
        """
        self.solver = self.get_solver(**kwargs)
        self.solver.build(self.graph, **kwargs)

    @monitor.time
    def train(self, **kwargs):
        """train(**kwargs)
        Train embeddings with the solver.
        Arguments depend on the underlying solver type.
        """
        self.solver.train(**kwargs)

    @monitor.time
    def evaluate(self, task, **kwargs):
        """evaluate(task, **kwargs)
        Evaluate the learned embeddings on a downstream task.
        Arguments depend on the underlying graph type and the task.

        Parameters:
            task (str): name of task

        Returns:
            dict: metrics and their values
        """
        func_name = task.replace(" ", "_")
        if not hasattr(self, func_name):
            raise ValueError("Unknown task `%s`" % task)

        logger.info(lib.io.header(task))
        result = getattr(self, func_name)(**kwargs)
        if isinstance(result, dict):
            for metric, value in sorted(result.items()):
                logger.warning("%s: %g" % (metric, value))

        return result

    @monitor.time
    def load_model(self, file_name):
        """
        Load model in pickle format.

        Parameters:
            file_name (str): file name:
        """
        logger.warning("load model from `%s`" % file_name)

        with open(file_name, "rb") as fin:
            model = pickle.load(fin)
        self.set_parameters(model)

    @monitor.time
    def save_model(self, file_name, save_hyperparameter=False):
        """
        Save model in pickle format.

        Parameters:
            file_name (str): file name
            save_hyperparameter (bool, optional): save hyperparameters or not, default is false
        """
        def is_mapping(name, attribute):
            return "2" in name

        def is_embedding(name, attribute):
            if name[0] == "_":
                return False
            return isinstance(attribute, np.ndarray)

        def is_hyperparameter(name, attribute):
            if name[0] == "_":
                return False
            return isinstance(attribute, int) or isinstance(attribute, float) or isinstance(attribute, str)

        def get_attributes(object, filter):
            attributes = EasyDict()
            for name in dir(object):
                attribute = getattr(object, name)
                if filter(name, attribute):
                    attributes[name] = attribute
            return attributes

        logger.warning("save model to `%s`" % file_name)

        model = EasyDict()
        model.graph = get_attributes(self.graph, is_mapping)
        model.solver = get_attributes(self.solver, is_embedding)
        if save_hyperparameter:
            model.graph.update(get_attributes(self.graph, is_hyperparameter))
            model.solver.update(get_attributes(self.solver, is_hyperparameter))
            model.solver.optimizer = get_attributes(self.solver.optimizer, is_hyperparameter)
            model.solver.optimizer.schedule = self.solver.optimizer.schedule.type

        with open(file_name, "wb") as fout:
            pickle.dump(model, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def get_mapping(self, id2name, name2id):
        mapping = []
        for name in id2name:
            if name not in name2id:
                raise ValueError("Can't find the embedding for `%s`" % name)
            mapping.append(name2id[name])
        return mapping

    def tokenize(self, str):
        str = str.strip(self.delimiters)
        comment_start = str.find(self.comment)
        if comment_start != -1:
            str = str[:comment_start]
        return self.pattern.split(str)

    def name_map(self, dicts, names):
        assert len(dicts) == len(names), "The number of dictionaries and names must be equal"

        indexes = [[] for _ in range(len(names))]
        num_param = len(names)
        num_sample = len(names[0])
        for i in range(num_sample):
            valid = True
            for j in range(num_param):
                if names[j][i] not in dicts[j]:
                    valid = False
                    break
            if valid:
                for j in range(num_param):
                    indexes[j].append(dicts[j][names[j][i]])
        return indexes

    def gpu_map(self, func, settings):
        import torch

        gpus = self.gpus if self.gpus else range(torch.cuda.device_count())
        new_settings = []
        for i, setting in enumerate(settings):
            new_settings.append(setting + (gpus[i % len(gpus)],))
        settings = new_settings

        try:
            start_method = multiprocessing.get_start_method()
            # if there are other running processes, this could cause leakage of semaphores
            multiprocessing.set_start_method("spawn", force=True)
            pool = multiprocessing.Pool(len(gpus))
            results = pool.map(func, settings, chunksize=1)
            multiprocessing.set_start_method(start_method, force=True)
        except AttributeError:
            logger.info("Spawn mode is not supported by multiprocessing. Switch to serial execution.")
            results = list(map(func, settings))

        return results


class GraphApplication(ApplicationMixin):
    """
    Node embedding application.

    Given a graph, it embeds each node into a continuous vector representation.
    The learned embeddings can be used for many downstream tasks.
    e.g. **node classification**, **link prediction**, **node analogy**.
    The similarity between node embeddings can be measured by cosine distance.

    Supported Models:
        - DeepWalk (`DeepWalk: Online Learning of Social Representations`_)
        - LINE (`LINE: Large-scale Information Network Embedding`_)
        - node2vec (`node2vec: Scalable Feature Learning for Networks`_)

    .. _DeepWalk\: Online Learning of Social Representations:
        https://arxiv.org/pdf/1403.6652.pdf
    .. _LINE\: Large-scale Information Network Embedding:
        https://arxiv.org/pdf/1503.03578.pdf
    .. _node2vec\: Scalable Feature Learning for Networks:
        https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf

    Parameters:
        dim (int): dimension of embeddings
        gpus (list of int, optional): GPU ids, default is all GPUs
        cpu_per_gpu (int, optional): number of CPU threads per GPU, default is all CPUs
        float_type (dtype, optional): type of parameters
        index_type (dtype, optional): type of graph indexes

    See also:
        :class:`Graph <graphvite.graph.Graph>`,
        :class:`GraphSolver <graphvite.solver.GraphSolver>`
    """

    def get_graph(self, **kwargs):
        return graph.Graph(self.index_type)

    def get_solver(self, **kwargs):
        if self.cpu_per_gpu == auto:
            num_sampler_per_worker = auto
        else:
            num_sampler_per_worker = self.cpu_per_gpu - 1
        return solver.GraphSolver(self.dim, self.float_type, self.index_type, self.gpus, num_sampler_per_worker,
                                  self.gpu_memory_limit)

    def set_parameters(self, model):
        mapping = self.get_mapping(self.graph.id2name, model.graph.name2id)
        self.solver.vertex_embeddings[:] = model.solver.vertex_embeddings[mapping]
        self.solver.context_embeddings[:] = model.solver.context_embeddings[mapping]

    def node_classification(self, X=None, Y=None, file_name=None, portions=(0.02,), normalization=False, times=1,
                            patience=100):
        """
        Evaluate node embeddings on node classification task.

        Parameters:
            X (list of str, optional): names of nodes
            Y (list, optional): labels of nodes
            file_name (str, optional): file of nodes & labels
            portions (tuple of float, optional): how much data for training
            normalization (bool, optional): normalize the embeddings or not
            times (int, optional): number of trials
            patience (int, optional): patience on loss convergence

        Returns:
            dict: macro-F1 & micro-F1 averaged over all trials
        """
        import scipy.sparse as sp

        self.solver.clear()

        if file_name:
            if not (X is None and Y is None):
                raise ValueError("Evaluation data and file should not be provided at the same time")
            X = []
            Y = []
            with open(file_name, "r") as fin:
                for line in fin:
                    tokens = self.tokenize(line)
                    if len(tokens) == 0:
                        continue
                    x, y = tokens
                    X.append(x)
                    Y.append(y)
        if X is None or Y is None:
            raise ValueError("Either evaluataion data (X, Y) or a file name should be provided")

        name2id = self.graph.name2id
        class2id = {c:i for i, c in enumerate(np.unique(Y))}
        new_X, new_Y = self.name_map((name2id, class2id), (X, Y))
        logger.info("effective labels: %d / %d" % (len(new_X), len(X)))
        X = np.asarray(new_X)
        Y = np.asarray(new_Y)

        labels = sp.coo_matrix((np.ones_like(X), (X, Y)), dtype=np.int32).todense()
        indexes, _ = np.where(np.sum(labels, axis=1) > 0)
        # discard non-labeled nodes
        labels = labels[indexes]
        vertex_embeddings = SharedNDArray(self.solver.vertex_embeddings[indexes])

        settings = []
        for portion in portions:
            settings.append((vertex_embeddings, labels, portion, normalization, times, patience))
        results = self.gpu_map(linear_classification, settings)

        metrics = {}
        for result in results:
            metrics.update(result)
        return metrics

    def link_prediction(self, H=None, T=None, Y=None, file_name=None, filter_H=None, filter_T=None, filter_file=None):
        """
        Evaluate node embeddings on link prediction task.

        Parameters:
            H (list of str, optional): names of head nodes
            T (list of str, optional): names of tail nodes
            Y (list of int, optional): labels of edges
            file_name (str, optional): file of edges and labels (e.g. validation set)
            filter_H (list of str, optional): names of head nodes to filter out
            filter_T (list of str, optional): names of tail nodes to filter out
            filter_file (str, optional): file of edges to filter out (e.g. training set)

        Returns:
            dict: AUC of link prediction
        """
        import torch

        from .network import LinkPredictor

        self.solver.clear()

        if file_name:
            if not (H is None and T is None and Y is None):
                raise ValueError("Evaluation data and file should not be provided at the same time")
            H = []
            T = []
            Y = []
            with open(file_name, "r") as fin:
                for line in fin:
                    tokens = self.tokenize(line)
                    if len(tokens) == 0:
                        continue
                    h, t, y = tokens
                    H.append(h)
                    T.append(t)
                    Y.append(y)
        if H is None or T is None or Y is None:
            raise ValueError("Either evaluation data or file should be provided")

        if filter_file:
            if not (filter_H is None and filter_T is None):
                raise ValueError("Filter data and file should not be provided at the same time")
            filter_H = []
            filter_T = []
            with open(filter_file, "r") as fin:
                for line in fin:
                    tokens = self.tokenize(line)
                    if len(tokens) == 0:
                        continue
                    h, t = tokens
                    filter_H.append(h)
                    filter_T.append(t)
        elif filter_H is None:
            filter_H = []
            filter_T = []

        name2id = self.graph.name2id
        Y = [int(y) for y in Y]
        new_H, new_T, new_Y = self.name_map((name2id, name2id, {0:0, 1:1}), (H, T, Y))
        logger.info("effective edges: %d / %d" % (len(new_H), len(H)))
        H = new_H
        T = new_T
        Y = new_Y
        new_H, new_T = self.name_map((name2id, name2id), (filter_H, filter_T))
        logger.info("effective filter edges: %d / %d" % (len(new_H), len(filter_H)))
        filters = set(zip(new_H, new_T))
        new_H = []
        new_T = []
        new_Y = []
        for h, t, y in zip(H, T, Y):
            if (h, t) not in filters:
                new_H.append(h)
                new_T.append(t)
                new_Y.append(y)
        logger.info("remaining edges: %d / %d" % (len(new_H), len(H)))
        H = np.asarray(new_H)
        T = np.asarray(new_T)
        Y = np.asarray(new_Y)

        vertex_embeddings = self.solver.vertex_embeddings
        context_embeddings = self.solver.context_embeddings
        model = LinkPredictor(self.solver.model, vertex_embeddings, context_embeddings)
        model = model.cuda()

        H = torch.as_tensor(H)
        T = torch.as_tensor(T)
        Y = torch.as_tensor(Y)
        H = H.cuda()
        T = T.cuda()
        Y = Y.cuda()
        score = model(H, T)
        order = torch.argsort(score, descending=True)
        Y = Y[order]
        hit = torch.cumsum(Y, dim=0)
        all = torch.sum(Y == 0) * torch.sum(Y == 1)
        auc = torch.sum(hit[Y == 0]).item() / all.item()

        return {
            "AUC": auc
        }


def linear_classification(args):
    import torch
    from torch import optim
    from torch.nn import functional as F
    from .network import NodeClassifier

    def generate_one_vs_rest(indexes, labels):
        new_indexes = []
        new_labels = []
        num_class = labels.shape[1]
        for index, sample_labels in zip(indexes, labels):
            for cls in np.where(sample_labels)[0]:
                new_indexes.append(index)
                new_label = np.zeros(num_class, dtype=np.int)
                new_label[cls] = 1
                new_labels.append(new_label)
        return torch.as_tensor(new_indexes), torch.as_tensor(new_labels)

    embeddings, labels, portion, normalization, times, patience, gpu = args
    num_sample, num_class = labels.shape
    num_train = int(num_sample * portion)

    macro_f1s = []
    micro_f1s = []
    for t in range(times):
        samples = np.random.permutation(num_sample)
        train_samples = samples[:num_train]
        train_labels = np.asarray(labels[train_samples])
        train_samples, train_labels = generate_one_vs_rest(train_samples, train_labels)
        test_samples = torch.as_tensor(samples[num_train:])
        test_labels = torch.as_tensor(labels[test_samples])

        model = NodeClassifier(embeddings, num_class, normalization=normalization)

        train_samples = train_samples.cuda(gpu)
        train_labels = train_labels.cuda(gpu)
        test_samples = test_samples.cuda(gpu)
        test_labels = test_labels.cuda(gpu)
        model = model.cuda(gpu)

        # train
        optimizer = optim.SGD(model.parameters(), lr=1, weight_decay=2e-5, momentum=0.9)
        best_loss = float("inf")
        best_epoch = -1
        for epoch in range(100000):
            optimizer.zero_grad()
            logits = model(train_samples)
            loss = F.binary_cross_entropy_with_logits(logits, train_labels.float())
            loss.backward()
            optimizer.step()

            loss = loss.item()
            if loss < best_loss:
                best_epoch = epoch
                best_loss = loss
            if epoch == best_epoch + patience:
                break

        # test
        logits = model(test_samples)
        num_labels = test_labels.sum(dim=1, keepdim=True)
        sorted, _ = logits.sort(dim=1, descending=True)
        thresholds = sorted.gather(dim=1, index=num_labels-1)
        predictions = (logits >= thresholds).int()
        # compute metric
        num_TP_per_class = (predictions & test_labels).sum(dim=0).float()
        num_T_per_class = test_labels.sum(dim=0).float()
        num_P_per_class = predictions.sum(dim=0).float()
        macro_f1s.append((2 * num_TP_per_class / (num_T_per_class + num_P_per_class)).mean().item())
        num_TP = (predictions & test_labels).sum().float()
        num_T = test_labels.sum().float()
        num_P = predictions.sum().float()
        micro_f1s.append((2 * num_TP / (num_T + num_P)).item())

    return {
        "macro-F1@%g%%" % (portion * 100): np.mean(macro_f1s),
        "micro-F1@%g%%" % (portion * 100): np.mean(micro_f1s)
    }


class WordGraphApplication(ApplicationMixin):
    """
    Word node embedding application.

    Given a corpus, it embeds each word into a continuous vector representation.
    The learned embeddings can be used for natural language processing tasks.
    This can be viewed as a variant of the word2vec algorithm, with random walk augmentation support.
    The similarity between node embeddings can be measured by cosine distance.

    Supported Models:
        - LINE (`LINE: Large-scale Information Network Embedding`_)

    Parameters:
        dim (int): dimension of embeddings
        gpus (list of int, optional): GPU ids, default is all GPUs
        cpu_per_gpu (int, optional): number of CPU threads per GPU, default is all CPUs
        float_type (dtype, optional): type of parameters
        index_type (dtype, optional): type of graph indexes

    See also:
        :class:`WordGraph <graphvite.graph.WordGraph>`,
        :class:`GraphSolver <graphvite.solver.GraphSolver>`
    """
    def get_graph(self, **kwargs):
        return graph.WordGraph(self.index_type)

    def get_solver(self, **kwargs):
        if self.cpu_per_gpu == auto:
            num_sampler_per_worker = auto
        else:
            num_sampler_per_worker = self.cpu_per_gpu - 1
        return solver.GraphSolver(self.dim, self.float_type, self.index_type, self.gpus, num_sampler_per_worker,
                                  self.gpu_memory_limit)

    def set_parameters(self, model):
        mapping = self.get_mapping(self.graph.id2name, model.graph.name2id)
        self.solver.vertex_embeddings[:] = model.solver.vertex_embeddings[mapping]
        self.solver.context_embeddings[:] = model.solver.context_embeddings[mapping]


class KnowledgeGraphApplication(ApplicationMixin):
    """
    Knowledge graph embedding application.

    Given a knowledge graph, it embeds each entity and relation into a continuous vector representation respectively.
    The learned embeddings can be used for analysis of knowledge graphs.
    e.g. **entity prediction**, **link prediction**.
    The likelihood of edges can be predicted by computing the score function over embeddings of triplets.

    Supported Models:
        - TransE (`Translating Embeddings for Modeling Multi-relational Data`_)
        - DistMult (`Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_)
        - ComplEx (`Complex Embeddings for Simple Link Prediction`_)
        - SimplE (`SimplE Embedding for Link Prediction in Knowledge Graphs`_)
        - RotatE (`RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_)
        - QuatE (`Quaternion Knowledge Graph Embeddings`)

    .. _Translating Embeddings for Modeling Multi-relational Data:
        http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf
    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf
    .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
        https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    .. _RotatE\: Knowledge Graph Embedding by Relational Rotation in Complex Space:
        https://arxiv.org/pdf/1902.10197.pdf
    .. _Quaternion Knowledge Graph Embeddings:
        https://papers.nips.cc/paper/8541-quaternion-knowledge-graph-embeddings.pdf

    Parameters:
        dim (int): dimension of embeddings
        gpus (list of int, optional): GPU ids, default is all GPUs
        cpu_per_gpu (int, optional): number of CPU threads per GPU, default is all CPUs
        float_type (dtype, optional): type of parameters
        index_type (dtype, optional): type of graph indexes

    Note:
        The implementation of TransE, DistMult, ComplEx, SimplE and QuatE are slightly different from their original
        papers.
        The loss function and the regularization term generally follow `this repo`_.
        Self-adversarial negative sampling is also adopted in these models like RotatE.

    .. _this repo: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

    See also:
        :class:`KnowledgeGraph <graphvite.graph.KnowledgeGraph>`,
        :class:`KnowledgeGraphSolver <graphvite.solver.KnowledgeGraphSolver>`
    """

    SAMPLE_PER_DIMENSION = 7
    MEMORY_SCALE_FACTOR = 1.5

    def get_graph(self, **kwargs):
        return graph.KnowledgeGraph(self.index_type)

    def get_solver(self, **kwargs):
        if self.cpu_per_gpu == auto:
            num_sampler_per_worker = auto
        else:
            num_sampler_per_worker = self.cpu_per_gpu - 1
        return solver.KnowledgeGraphSolver(self.dim, self.float_type, self.index_type, self.gpus, num_sampler_per_worker,
                                           self.gpu_memory_limit)

    def set_parameters(self, model):
        entity_mapping = self.get_mapping(self.graph.id2entity, model.graph.entity2id)
        relation_mapping = self.get_mapping(self.graph.id2relation, model.graph.relation2id)
        self.solver.entity_embeddings[:] = model.solver.entity_embeddings[entity_mapping]
        self.solver.relation_embeddings[:] = model.solver.relation_embeddings[relation_mapping]

    def entity_prediction(self, H=None, R=None, T=None, file_name=None, save_file=None, target="tail", k=10,
                          backend=cfg.backend):
        """
        Predict the distribution of missing entity or relation for triplets.

        Parameters:
            H (list of str, optional): names of head entities
            R (list of str, optional): names of relations
            T (list of str, optional): names of tail entities
            file_name (str, optional): file of triplets (e.g. validation set)
            save_file (str, optional): ``txt`` or ``pkl`` file to save predictions
            k (int, optional): top-k recalls will be returned
            target (str, optional): 'head' or 'tail'
            backend (str, optional): 'graphvite' or 'torch'

        Return:
            list of list of tuple: top-k recalls for each triplet, if save file is not provided
        """
        def torch_predict():
            import torch

            entity_embeddings = SharedNDArray(self.solver.entity_embeddings)
            relation_embeddings = SharedNDArray(self.solver.relation_embeddings)

            num_gpu = len(self.gpus) if self.gpus else torch.cuda.device_count()
            work_load = (num_sample + num_gpu - 1) // num_gpu
            settings = []

            for i in range(num_gpu):
                work_H = H[work_load * i: work_load * (i+1)]
                work_R = R[work_load * i: work_load * (i+1)]
                work_T = T[work_load * i: work_load * (i+1)]
                settings.append((entity_embeddings, relation_embeddings, work_H, work_R, work_T,
                                 None, None, target, k, self.solver.model, self.solver.margin))

            results = self.gpu_map(triplet_prediction, settings)
            return sum(results, [])

        def graphvite_predict():
            num_entity = len(entity2id)
            batch_size = self.get_batch_size(num_entity)
            recalls = []

            for i in range(0, num_sample, batch_size):
                batch_h = H[i: i + batch_size]
                batch_r = R[i: i + batch_size]
                batch_t = T[i: i + batch_size]
                batch = self.generate_one_vs_rest(batch_h, batch_r, batch_t, num_entity, target)

                scores = self.solver.predict(batch)
                scores = scores.reshape(-1, num_entity)
                indexes = np.argpartition(scores, num_entity - k, axis=-1)
                for index, score in zip(indexes, scores):
                    index = index[-k:]
                    score = score[index]
                    order = np.argsort(score)[::-1]
                    recall = list(zip(index[order], score[order]))
                    recalls.append(recall)

            return recalls

        assert_in(["head", "tail"], target=target)
        assert_in(["graphvite", "torch"], backend=backend)

        if backend == "torch":
            self.solver.clear()

        if file_name:
            if not (H is None and R is None and T is None):
                raise ValueError("Evaluation data and file should not be provided at the same time")
            H = []
            R = []
            T = []
            with open(file_name, "r") as fin:
                for i, line in enumerate(fin):
                    tokens = self.tokenize(line)
                    if len(tokens) == 0:
                        continue
                    if 3 <= len(tokens) <= 4:
                        h, r, t = tokens[:3]
                    elif len(tokens) == 2:
                        if target == "head":
                            r, t = tokens
                            h = None
                        else:
                            h, r = tokens
                            t = None
                    else:
                        raise ValueError("Invalid line format at line %d in %s" % (i + 1, file_name))
                    H.append(h)
                    R.append(r)
                    T.append(t)
        if (H is None and T is None) or R is None:
            raise ValueError("Either evaluation data or file should be provided")
        if H is None:
            target = "head"
        if T is None:
            target = "tail"

        entity2id = self.graph.entity2id
        relation2id = self.graph.relation2id
        num_sample = len(R)
        new_H = np.zeros(num_sample, dtype=np.uint32)
        new_T = np.zeros(num_sample, dtype=np.uint32)
        if target == "head":
            new_R, new_T = self.name_map((relation2id, entity2id), (R, T))
        if target == "tail":
            new_H, new_R = self.name_map((entity2id, relation2id), (H, R))
        assert len(new_R) == len(R), "Can't recognize some entities or relations"
        H = np.asarray(new_H, dtype=np.uint32)
        R = np.asarray(new_R, dtype=np.uint32)
        T = np.asarray(new_T, dtype=np.uint32)

        if backend == "graphvite":
            recalls = graphvite_predict()
        else:
            recalls = torch_predict()

        id2entity = self.graph.id2entity
        new_recalls = []
        for recall in recalls:
            new_recall = [(id2entity[e], s) for e, s in recall]
            new_recalls.append(new_recall)
        recalls = new_recalls

        if save_file:
            logger.warning("save entity predictions to `%s`" % save_file)
            extension = os.path.splitext(save_file)[1]
            if extension == ".txt":
                with open(save_file, "w") as fout:
                    for recall in recalls:
                        tokens = ["%s: %g" % x for x in recall]
                        fout.write("%s\n" % "\t".join(tokens))
            elif extension == ".pkl":
                with open(save_file, "wb") as fout:
                    pickle.dump(recalls, fout, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError("Unknown file extension `%s`" % extension)
        else:
            return recalls

    def link_prediction(self, H=None, R=None, T=None, filter_H=None, filter_R=None, filter_T=None, file_name=None,
                        filter_files=None, target="both", fast_mode=None, backend=cfg.backend):
        """
        Evaluate knowledge graph embeddings on link prediction task.

        Parameters:
            H (list of str, optional): names of head entities
            R (list of str, optional): names of relations
            T (list of str, optional): names of tail entities
            file_name (str, optional): file of triplets (e.g. validation set)
            filter_H (list of str, optional): names of head entities to filter out
            filter_R (list of str, optional): names of relations to filter out
            filter_T (list of str, optional): names of tail entities to filter out
            filter_files (str, optional): files of triplets to filter out (e.g. training / validation / test set)
            target (str, optional): 'head', 'tail' or 'both'
            fast_mode (int, optional): if specified, only that number of samples will be evaluated
            backend (str, optional): 'graphvite' or 'torch'

        Returns:
            dict: MR, MRR, HITS\@1, HITS\@3 & HITS\@10 of link prediction
        """
        def torch_predict():
            import torch

            entity_embeddings = SharedNDArray(self.solver.entity_embeddings)
            relation_embeddings = SharedNDArray(self.solver.relation_embeddings)

            num_gpu = len(self.gpus) if self.gpus else torch.cuda.device_count()
            work_load = (fast_mode + num_gpu - 1) // num_gpu
            settings = []

            for i in range(num_gpu):
                work_H = H[work_load * i: work_load * (i+1)]
                work_R = R[work_load * i: work_load * (i+1)]
                work_T = T[work_load * i: work_load * (i+1)]
                settings.append((entity_embeddings, relation_embeddings, work_H, work_R, work_T,
                                 exclude_H, exclude_T, target, None, self.solver.model, self.solver.margin))

            results = self.gpu_map(triplet_prediction, settings)
            return np.concatenate(results)

        def graphvite_predict():
            num_entity = len(entity2id)
            if target == "both":
                batch_size = self.get_batch_size(num_entity * 2)
            else:
                batch_size = self.get_batch_size(num_entity)
            rankings = []

            for i in range(0, fast_mode, batch_size):
                batch_h = H[i: i + batch_size]
                batch_r = R[i: i + batch_size]
                batch_t = T[i: i + batch_size]
                batch = self.generate_one_vs_rest(batch_h, batch_r, batch_t, num_entity, target)
                masks = self.generate_mask(batch_h, batch_r, batch_t, exclude_H, exclude_T, num_entity, target)
                if target == "head":
                    positives = batch_h
                if target == "tail":
                    positives = batch_t
                if target == "both":
                    positives = np.asarray([batch_h, batch_t]).transpose()
                    positives = positives.ravel()

                scores = self.solver.predict(batch)
                scores = scores.reshape(-1, num_entity)
                truths = scores[range(len(positives)), positives]
                ranking = np.sum((scores >= truths[:, np.newaxis]) * masks, axis=1)
                rankings.append(ranking)

            return np.concatenate(rankings)

        assert_in(["head", "tail", "both"], target=target)
        assert_in(["graphvite", "torch"], backend=backend)

        if backend == "torch":
            self.solver.clear()

        if file_name:
            if not (H is None and R is None and T is None):
                raise ValueError("Evaluation data and file should not be provided at the same time")
            H = []
            R = []
            T = []
            with open(file_name, "r") as fin:
                for i, line in enumerate(fin):
                    tokens = self.tokenize(line)
                    if len(tokens) == 0:
                        continue
                    if 3 <= len(tokens) <= 4:
                        h, r, t = tokens[:3]
                    else:
                        raise ValueError("Invalid line format at line %d in %s" % (i + 1, file_name))
                    H.append(h)
                    R.append(r)
                    T.append(t)
        if H is None or R is None or T is None:
            raise ValueError("Either evaluation data or file should be provided")

        if filter_files:
            if not (filter_H is None and filter_R is None and filter_T is None):
                raise ValueError("Filter data and file should not be provided at the same time")
            filter_H = []
            filter_R = []
            filter_T = []
            for filter_file in filter_files:
                with open(filter_file, "r") as fin:
                    for i, line in enumerate(fin):
                        tokens = self.tokenize(line)
                        if len(tokens) == 0:
                            continue
                        if 3 <= len(tokens) <= 4:
                            h, r, t = tokens[:3]
                        else:
                            raise ValueError("Invalid line format at line %d in %s" % (i + 1, filter_file))
                        filter_H.append(h)
                        filter_R.append(r)
                        filter_T.append(t)
        elif filter_H is None:
            filter_H = []
            filter_R = []
            filter_T = []

        entity2id = self.graph.entity2id
        relation2id = self.graph.relation2id
        new_H, new_R, new_T = self.name_map((entity2id, relation2id, entity2id), (H, R, T))
        logger.info("effective triplets: %d / %d" % (len(new_H), len(H)))
        H = np.asarray(new_H, dtype=np.uint32)
        R = np.asarray(new_R, dtype=np.uint32)
        T = np.asarray(new_T, dtype=np.uint32)
        new_H, new_R, new_T = self.name_map((entity2id, relation2id, entity2id), (filter_H, filter_R, filter_T))
        logger.info("effective filter triplets: %d / %d" % (len(new_H), len(filter_H)))
        filter_H = np.asarray(new_H, dtype=np.uint32)
        filter_R = np.asarray(new_R, dtype=np.uint32)
        filter_T = np.asarray(new_T, dtype=np.uint32)

        exclude_H = defaultdict(set)
        exclude_T = defaultdict(set)
        for h, r, t in zip(filter_H, filter_R, filter_T):
            exclude_H[(t, r)].add(h)
            exclude_T[(h, r)].add(t)

        num_sample = len(H)
        fast_mode = fast_mode or num_sample
        indexes = np.random.permutation(num_sample)[:fast_mode]
        H = H[indexes]
        R = R[indexes]
        T = T[indexes]

        if backend == "graphvite":
            rankings = graphvite_predict()
        elif backend == "torch":
            rankings = torch_predict()

        return {
            "MR": np.mean(rankings),
            "MRR": np.mean(1 / rankings),
            "HITS@1": np.mean(rankings <= 1),
            "HITS@3": np.mean(rankings <= 3),
            "HITS@10": np.mean(rankings <= 10)
        }

    def get_batch_size(self, sample_size):
        import psutil
        memory = psutil.virtual_memory()

        batch_size = int(self.SAMPLE_PER_DIMENSION * self.dim * self.graph.num_vertex
                         * self.solver.num_partition / self.solver.num_worker / sample_size)
        # 2 triplet (Python, C++ sample pool) + 1 sample index
        mem_per_sample = sample_size * (2 * 3 * np.uint32().itemsize + 1 * np.uint64().itemsize)
        max_batch_size = int(memory.available / mem_per_sample / self.MEMORY_SCALE_FACTOR)
        if max_batch_size < batch_size:
            logger.info("Memory is not enough for optimal prediction batch size. "
                        "Use the maximal possible size instead.")
            batch_size = max_batch_size
        return batch_size

    def generate_one_vs_rest(self, H, R, T, num_entity, target="both"):
        one = np.ones(num_entity, dtype=np.bool)
        all = np.arange(num_entity, dtype=np.uint32)
        batches = []

        for h, r, t in zip(H, R, T):
            if target == "head" or target == "both":
                batch = np.asarray([all, t * one, r * one]).transpose()
                batches.append(batch)
            if target == "tail" or target == "both":
                batch = np.asarray([h * one, all, r * one]).transpose()
                batches.append(batch)

        batches = np.concatenate(batches)
        return batches

    def generate_mask(self, H, R, T, exclude_H, exclude_T, num_entity, target="both"):
        one = np.ones(num_entity, dtype=np.bool)
        masks = []

        for h, r, t in zip(H, R, T):
            if target == "head" or target == "both":
                mask = one.copy()
                mask[list(exclude_H[(t, r)])] = 0
                mask[h] = 1
                masks.append(mask)
            if target == "tail" or target == "both":
                mask = one.copy()
                mask[list(exclude_T[(h, r)])] = 0
                mask[t] = 1
                masks.append(mask)

        masks = np.asarray(masks)
        return masks


def triplet_prediction(args):
    import torch
    from .network import LinkPredictor
    torch.set_grad_enabled(False)

    entity_embeddings, relation_embeddings, H, R, T, \
    exclude_H, exclude_T, target, k, score_function, margin, device = args
    num_entity = len(entity_embeddings)
    score_function = LinkPredictor(score_function, entity_embeddings, relation_embeddings, entity_embeddings,
                                   margin=margin)

    if device != "cpu":
        try:
            score_function = score_function.to(device)
        except RuntimeError:
            logger.info("Model is too large for GPU evaluation with PyTorch. Switch to CPU evaluation.")
            device = "cpu"
        if device == "cpu":
            del score_function
            torch.cuda.empty_cache()
            score_function = LinkPredictor(score_function, entity_embeddings, relation_embeddings, entity_embeddings,
                                           margin=margin)

    one = torch.ones(num_entity, dtype=torch.long, device=device)
    all = torch.arange(num_entity, dtype=torch.long, device=device)
    results = [] # rankings or top-k recalls

    for h, r, t in zip(H, R, T):
        if target == "head" or target == "both":
            batch_h = all
            batch_r = r * one
            batch_t = t * one
            score = score_function(batch_h, batch_r, batch_t)
            if k: # top-k recalls
                score, index = torch.topk(score, k)
                score = score.cpu().numpy()
                index = index.cpu().numpy()
                recall = list(zip(index, score))
                results.append(recall)
            else: # ranking
                mask = torch.ones(num_entity, dtype=torch.uint8, device=device)
                index = torch.tensor(list(exclude_H[(t, r)]), dtype=torch.long, device=device)
                mask[index] = 0
                mask[h] = 1
                ranking = torch.sum((score >= score[h]) * mask).item()
                results.append(ranking)

        if target == "tail" or target == "both":
            batch_h = h * one
            batch_r = r * one
            batch_t = all
            score = score_function(batch_h, batch_r, batch_t)
            if k: # top-k recalls
                score, index = torch.topk(score, k)
                score = score.cpu().numpy()
                index = index.cpu().numpy()
                recall = list(zip(index, score))
                results.append(recall)
            else: # ranking
                mask = torch.ones(num_entity, dtype=torch.uint8, device=device)
                index = torch.tensor(list(exclude_T[(h, r)]), dtype=torch.long, device=device)
                mask[index] = 0
                mask[t] = 1
                ranking = torch.sum((score >= score[t]) * mask).item()
                results.append(ranking)

    if not k: # ranking
        results = np.asarray(results)
    return results


class VisualizationApplication(ApplicationMixin):
    """
    Graph & high-dimensional data visualization.
    
    Given a graph or high-dimensional vectors, it maps each node to 2D or 3D coordinates to
    faciliate visualization. The learned coordinates preserve most local similarity information
    of the original input, and may shed some light on the structure of the graph or the
    high-dimensional space.

    Supported Models:
        - LargeVis (`Visualizing Large-scale and High-dimensional Data`_)

    .. _Visualizing Large-scale and High-dimensional Data: https://arxiv.org/pdf/1602.00370.pdf

    Parameters:
        dim (int): dimension of embeddings
        gpus (list of int, optional): GPU ids, default is all GPUs
        cpu_per_gpu (int, optional): number of CPU threads per GPU, default is all CPUs
        float_type (dtype, optional): type of parameters
        index_type (dtype, optional): type of graph indexes

    See also:
        :class:`Graph <graphvite.graph.Graph>`,
        :class:`KNNGraph <graphvite.graph.KNNGraph>`,
        :class:`VisualizationSolver <graphvite.solver.VisualizationSolver>`
    """

    OUTLIER_THRESHOLD = 5

    def get_graph(self, **kwargs):
        if "file_name" in kwargs or "edge_list" in kwargs:
            return graph.Graph(self.index_type)
        else:
            return graph.KNNGraph(self.index_type, self.gpus, self.cpu_per_gpu)

    def get_solver(self, **kwargs):
        if self.cpu_per_gpu == auto:
            num_sampler_per_worker = auto
        else:
            num_sampler_per_worker = self.cpu_per_gpu - 1

        return solver.VisualizationSolver(self.dim, self.float_type, self.index_type, self.gpus, num_sampler_per_worker,
                                          self.gpu_memory_limit)

    def set_parameters(self, model):
        if self.solver.coordinates.shape != model.solver.coordinates.shape:
            raise ValueError("Expect coordinates with shape %s, but %s is found" %
                             (self.solver.coordinates.shape, model.solver.coordinates.shape))
        self.solver.coordinates[:] = model.solver.coordinates

    def visualization(self, Y=None, file_name=None, save_file=None, figure_size=10, scale=2):
        """
        Visualize learned 2D or 3D coordinates.

        Parameters:
            Y (list of str, optional): labels of vectors
            file_name (str, optional): file of labels
            save_file (str, optional): ``png`` or ``pdf`` file to save visualization,
                if not provided, show the figure in window
            figure_size (int, optional): size of figure
            scale (int, optional): size of points
        """
        from matplotlib import pyplot as plt
        plt.switch_backend("agg") # for compatibility

        self.solver.clear()

        coordinates = self.solver.coordinates
        dim = coordinates.shape[1]
        if not (dim == 2 or dim == 3):
            raise ValueError("Can't visualize %dD data" % dim)

        if file_name:
            if not (Y is None):
                raise ValueError("Evaluation data and file should not be provided at the same time")
            Y = []
            with open(file_name, "r") as fin:
                for line in fin:
                    tokens = self.tokenize(line)
                    if len(tokens) == 0:
                        continue
                    y, = tokens
                    Y.append(y)
        elif Y is None:
            Y = ["unknown"] * self.graph.num_vertex
        Y = np.asarray(Y)

        mean = np.mean(coordinates, axis=0)
        std = np.std(coordinates, axis=0)
        inside = np.abs(coordinates - mean) < self.OUTLIER_THRESHOLD * std
        indexes, = np.where(np.all(inside, axis=1))
        # discard outliers
        coordinates = coordinates[indexes]
        Y = Y[indexes]
        classes = sorted(np.unique(Y))

        fig = plt.figure(figsize=(figure_size, figure_size))
        if dim == 2:
            ax = fig.gca()
        elif dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.gca(projection="3d")
        for cls in classes:
            indexes, = np.where(Y == cls)
            ax.scatter(*coordinates[indexes].T, s=scale)
        ax.set_xticks([])
        ax.set_yticks([])
        if dim == 3:
            ax.set_zticks([])
        if len(classes) > 1:
            ax.legend(classes, markerscale=6, loc="upper right")
        if save_file:
            logger.warning("save visualization to `%s`" % save_file)
            plt.savefig(save_file)
        else:
            plt.show()

        return {}

    def hierarchy(self, HY=None, file_name=None, target=None, save_file=None, figure_size=10, scale=2, duration=3):
        """
        Visualize learned 2D coordinates with hierarchical labels.

        Parameters:
            HY (list of list of str, optional): hierarchical labels of vectors
            file_name (str, optional): file of hierarchical labels
            target (str): target class
            save_file (str): ``gif`` file to save visualization
            figure_size (int, optional): size of figure
            scale (int, optional): size of points
            duration (float, optional): duration of each frame in seconds
        """
        import imageio
        from matplotlib import pyplot as plt
        plt.switch_backend("agg") # for compatibility

        self.solver.clear()

        coordinates = self.solver.coordinates
        dim = coordinates.shape[1]
        if dim != 2:
            raise ValuerError("Can't visualize the hierarchy of %dD data" % dim)

        if file_name:
            if not (HY is None):
                raise ValueError("Evaluation data and file should not be provided at the same time")
            HY = []
            with open(file_name, "r") as fin:
                for line in fin:
                    tokens = self.tokenize(line)
                    if len(tokens) > 0:
                        HY.append(tokens)
        elif HY is None:
            raise ValueError("No label is provided for hierarchy")
        HY = np.asarray(HY)
        min_type = "S%d" % len("else")
        if HY.dtype < min_type:
            HY = HY.astype(min_type)

        mean = np.mean(coordinates, axis=0)
        std = np.std(coordinates, axis=0)
        inside = np.abs(coordinates - mean) < self.OUTLIER_THRESHOLD * std
        indexes, = np.where(np.all(inside, axis=1))
        # discard outliers
        coordinates = coordinates[indexes]
        HY = HY[indexes].T

        if target is None:
            raise ValueError("Target class is not provided")
        for depth, Y in enumerate(HY):
            indexes, = np.where(Y == target)
            if len(indexes) > 0:
                sample = indexes[0]
                break
        else:
            raise ValueError("Can't find target `%s` in the hierarchy" % target)

        settings = [(coordinates, None, HY[0], sample, figure_size, scale, 0)]
        for i in range(depth):
            settings.append((coordinates, HY[i], HY[i + 1], sample, figure_size, scale, i+1))
        pool = multiprocessing.Pool(self.solver.num_worker + self.solver.num_sampler)
        frames = pool.map(render_hierarchy, settings)
        logger.warning("save hierarchy to `%s`" % save_file)
        imageio.mimsave(save_file, frames, fps=1 / duration, subrectangles=True)

        return {}

    def animation(self, Y=None, file_name=None, save_file=None, figure_size=5, scale=1, elevation=30, num_frame=700):
        """
        Rotate learn 3D coordinates as an animation.

        Parameters:
            Y (list of str, optional): labels of vectors
            file_name (str, optional): file of labels
            save_file (str): ``gif`` file to save visualization
            figure_size (int, optional): size of figure
            scale (int, optional): size of points
            elevation (float, optional): elevation angle
            num_frame (int, optional): number of frames
        """
        import imageio
        from matplotlib import pyplot as plt, animation
        from mpl_toolkits.mplot3d import Axes3D
        plt.switch_backend("agg") # for compatibility

        self.solver.clear()

        coordinates = self.solver.coordinates
        dim = coordinates.shape[1]
        if dim != 3:
            raise ValueError("Can't animate %dD data" % dim)

        if file_name:
            if not (Y is None):
                raise ValueError("Evaluation data and file should not be provided at the same time")
            Y = []
            with open(file_name, "r") as fin:
                for line in fin:
                    tokens = self.tokenize(line)
                    if len(tokens) == 0:
                        continue
                    y, = tokens
                    Y.append(y)
        elif Y is None:
            Y = ["unknown"] * self.graph.num_vertex
        Y = np.asarray(Y)

        mean = np.mean(coordinates, axis=0)
        std = np.std(coordinates, axis=0)
        inside = np.abs(coordinates - mean) < self.OUTLIER_THRESHOLD * std
        indexes, = np.where(np.all(inside, axis=1))
        # discard outliers
        coordinates = coordinates[indexes]
        Y = Y[indexes]

        settings = []
        degrees = np.linspace(0, 360, num_frame, endpoint=False)
        for degree in degrees:
            settings.append((coordinates, Y, degree, figure_size, scale, elevation))
        pool = multiprocessing.Pool(self.solver.num_worker + self.solver.num_sampler)
        frames = pool.map(render_animation, settings)
        logger.warning("save animation to `%s`" % save_file)
        imageio.mimsave(save_file, frames, fps=num_frame / 70, subrectangles=True) # 70 seconds

        return {}


def render_hierarchy(args):
    from matplotlib import pyplot as plt
    plt.switch_backend("agg")

    coordinates, H, Y, sample, figure_size, scale, depth = args

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.gca()
    if H is not None:
        for i in range(len(Y)):
            if H[i] != H[sample]:
                Y[i] = "else"
    classes = set(Y)
    classes.discard(Y[sample])
    classes.discard("else")
    classes = [Y[sample]] + sorted(classes) + ["else"]
    for i, cls in enumerate(classes):
        indexes, = np.where(Y == cls)
        color = "lightgrey" if cls == "else" else None
        ax.scatter(*coordinates[indexes].T, s=2, c=color, zorder=-i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(classes, markerscale=6, loc="upper right")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.renderer._renderer)

    return frame


def render_animation(args):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.switch_backend("agg")

    coordinates, Y, degree, figure_size, scale, elevation = args
    classes = sorted(np.unique(Y))

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.gca(projection="3d")
    for cls in classes:
        indexes, = np.where(Y == cls)
        ax.scatter(*coordinates[indexes].T, s=scale)
    ax.view_init(elev=elevation, azim=degree)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if len(classes) > 1:
        ax.legend(classes, markerscale=6)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.renderer._renderer)

    return frame


class Application(object):
    """
    Application(type, *args, **kwargs)
    Create an application instance of any type.

    Parameters:
        type (str): application type,
            can be 'graph', 'word graph', 'knowledge graph' or 'visualization'
    """

    application = {
        "graph": GraphApplication,
        "word graph": WordGraphApplication,
        "knowledge graph": KnowledgeGraphApplication,
        "visualization": VisualizationApplication
    }

    def __new__(cls, type, *args, **kwargs):
        if type in cls.application:
            return cls.application[type](*args, **kwargs)
        else:
            raise ValueError("Unknown application `%s`" % type)

__all__ = [
    "Application",
    "GraphApplication", "WordGraphApplication", "KnowledgeGraphApplication", "VisualizationApplication"
]