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

"""
Dataset module of GraphVite

Graph

- :class:`BlogCatalog`
- :class:`Youtube`
- :class:`Flickr`
- :class:`Hyperlink2012`
- :class:`Friendster`
- :class:`Wikipedia`

Knowledge Graph

- :class:`Math`
- :class:`FB15k`
- :class:`FB15k237`
- :class:`WN18`
- :class:`WN18RR`
- :class:`Wikidata5m`
- :class:`Freebase`

Visualization

- :class:`MNIST`
- :class:`CIFAR10`
- :class:`ImageNet`
"""
from __future__ import absolute_import, division

import os
import glob
import shutil
import logging
import gzip, zipfile, tarfile
import multiprocessing
from collections import defaultdict

import numpy as np

from . import cfg

logger = logging.getLogger(__name__)


class Dataset(object):
    """
    Graph dataset.

    Parameters:
        name (str): name of dataset
        urls (dict, optional): url(s) for each split,
            can be either str or list of str
        members (dict, optional): zip member(s) for each split,
            leave empty for default

    Datasets contain several splits, such as train, valid and test.
    For each split, there are one or more URLs, specifying the file to download.
    You may also specify the zip member to extract.
    When a split is accessed, it will be automatically downloaded and decompressed
    if it is not present.

    You can assign a preprocess for each split, by defining a function with name [split]_preprocess::

        class MyDataset(Dataset):
            def __init__(self):
                super(MyDataset, self).__init__(
                    "my_dataset",
                    train="url/to/train/split",
                    test="url/to/test/split"
                )

            def train_preprocess(self, input_file, output_file):
                with open(input_file, "r") as fin, open(output_file, "w") as fout:
                    fout.write(fin.read())

        f = open(MyDataset().train)

    If the preprocess returns a non-trivial value, then it is assigned to the split,
    otherwise the file name is assigned.
    By convention, only splits ending with ``_data`` have non-trivial return value.

    See also:
        Pre-defined preprocess functions
        :func:`csv2txt`,
        :func:`top_k_label`,
        :func:`induced_graph`,
        :func:`edge_split`,
        :func:`link_prediction_split`,
        :func:`image_feature_data`
    """
    def __init__(self, name, urls=None, members=None):
        self.name = name
        self.urls = urls or {}
        self.members = members or {}
        for key in self.urls:
            if isinstance(self.urls[key], str):
                self.urls[key] = [self.urls[key]]
            if key not in self.members:
                self.members[key] = [None] * len(self.urls[key])
            elif isinstance(self.members[key], str):
                self.members[key] = [self.members[key]]
            if len(self.urls[key]) != len(self.members[key]):
                raise ValueError("Number of members is inconsistent with number of urls in `%s`" % key)
        self.path = os.path.join(cfg.dataset_path, self.name)

    def relpath(self, path):
        return os.path.relpath(path, self.path)

    def download(self, url):
        from six.moves.urllib.request import urlretrieve

        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
        save_file = os.path.join(self.path, save_file)
        if save_file in self.local_files():
            return save_file

        logger.info("downloading %s to %s" % (url, self.relpath(save_file)))
        urlretrieve(url, save_file)
        return save_file

    def extract(self, zip_file, member=None):
        zip_name, extension = os.path.splitext(zip_file)
        if zip_name.endswith(".tar"):
            extension = ".tar" + extension
            zip_name = zip_name[:-4]

        if extension == ".txt":
            return zip_file
        elif member is None:
            save_file = zip_name
        else:
            save_file = os.path.join(os.path.dirname(zip_name), os.path.basename(member))
        if save_file in self.local_files():
            return save_file

        if extension == ".gz":
            logger.info("extracting %s to %s" % (self.relpath(zip_file), self.relpath(save_file)))
            with gzip.open(zip_file, "rb") as fin, open(save_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        elif extension == ".tar.gz" or extension == ".tar":
            if member is None:
                logger.info("extracting %s to %s" % (self.relpath(zip_file), self.relpath(save_file)))
                with tarfile.open(zip_file, "r") as fin:
                    fin.extractall(save_file)
            else:
                logger.info("extracting %s from %s to %s" % (member, self.relpath(zip_file), self.relpath(save_file)))
                with tarfile.open(zip_file, "r").extractfile(member) as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        elif extension == ".zip":
            if member is None:
                logger.info("extracting %s to %s" % (self.relpath(zip_file), self.relpath(save_file)))
                with zipfile.ZipFile(zip_file) as fin:
                    fin.extractall(save_file)
            else:
                logger.info("extracting %s from %s to %s" % (member, self.relpath(zip_file), self.relpath(save_file)))
                with zipfile.ZipFile(zip_file).open(member, "r") as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        else:
            raise ValueError("Unknown file extension `%s`" % extension)

        return save_file

    def get_file(self, key):
        file_name = os.path.join(self.path, "%s_%s.txt" % (self.name, key))
        if file_name in self.local_files():
            return file_name

        urls = self.urls[key]
        members = self.members[key]
        preprocess_name = key + "_preprocess"
        preprocess = getattr(self, preprocess_name, None)
        if len(urls) > 1 and preprocess is None:
            raise AttributeError(
                "There are non-trivial number of files, but function `%s` is not found" % preprocess_name)

        extract_files = []
        for url, member in zip(urls, members):
            download_file = self.download(url)
            extract_file = self.extract(download_file, member)
            extract_files.append(extract_file)
        if preprocess:
            result = preprocess(*(extract_files + [file_name]))
            if result is not None:
                return result
        elif os.path.isfile(extract_files[0]):
            logger.info("renaming %s to %s" % (self.relpath(extract_files[0]), self.relpath(file_name)))
            shutil.move(extract_files[0], file_name)
        else:
            raise AttributeError(
                "There are non-trivial number of files, but function `%s` is not found" % preprocess_name)

        return file_name

    def local_files(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        return set(glob.glob(os.path.join(self.path, "*")))

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self.urls:
            return self.get_file(key)
        raise AttributeError("Can't resolve split `%s`" % key)

    def csv2txt(self, csv_file, txt_file):
        """
        Convert ``csv`` to ``txt``.

        Parameters:
            csv_file: csv file
            txt_file: txt file
        """
        logger.info("converting %s to %s" % (self.relpath(csv_file), self.relpath(txt_file)))
        with open(csv_file, "r") as fin, open(txt_file, "w") as fout:
            for line in fin:
                fout.write(line.replace(",", "\t"))

    def top_k_label(self, label_file, save_file, k, format="node-label"):
        """
        Extract top-k labels.

        Parameters:
            label_file (str): label file
            save_file (str): save file
            k (int): top-k labels will be extracted
            format (str, optional): format of label file,
            can be 'node-label' or '(label)-nodes':
                - **node-label**: each line is [node] [label]
                - **(label)-nodes**: each line is [node]..., no explicit label
        """
        logger.info("extracting top-%d labels of %s to %s" % (k, self.relpath(label_file), self.relpath(save_file)))
        if format == "node-label":
            label2nodes = defaultdict(list)
            with open(label_file, "r") as fin:
                for line in fin:
                    node, label = line.split()
                    label2nodes[label].append(node)
        elif format == "(label)-nodes":
            label2nodes = {}
            with open(label_file, "r") as fin:
                for i, line in enumerate(fin):
                    label2nodes[i] = line.split()
        else:
            raise ValueError("Unknown file format `%s`" % format)

        labels = sorted(label2nodes, key=lambda x: len(label2nodes[x]), reverse=True)[:k]
        with open(save_file, "w") as fout:
            for label in sorted(labels):
                for node in sorted(label2nodes[label]):
                    fout.write("%s\t%s\n" % (node, label))

    def induced_graph(self, graph_file, label_file, save_file):
        """
        Induce a subgraph from labeled nodes. All edges in the induced graph have at least one labeled node.

        Parameters:
            graph_file (str): graph file
            label_file (str): label file
            save_file (str): save file
        """
        logger.info("extracting subgraph of %s induced by %s to %s" %
              (self.relpath(graph_file), self.relpath(label_file), self.relpath(save_file)))
        nodes = set()
        with open(label_file, "r") as fin:
            for line in fin:
                nodes.update(line.split())
        with open(graph_file, "r") as fin, open(save_file, "w") as fout:
            for line in fin:
                if not line.startswith("#"):
                    u, v = line.split()
                    if u not in nodes or v not in nodes:
                        continue
                    fout.write("%s\t%s\n" % (u, v))

    def edge_split(self, graph_file, files, portions):
        """
        Divide a graph into several splits.

        Parameters:
            graph_file (str): graph file
            files (list of str): file names
            portions (list of float): split portions
        """
        assert len(files) == len(portions)
        logger.info("splitting graph %s into %s" %
                    (self.relpath(graph_file), ", ".join([self.relpath(file) for file in files])))
        np.random.seed(1024)

        portions = np.cumsum(portions, dtype=np.float32) / np.sum(portions)
        files = [open(file, "w") for file in files]
        with open(graph_file, "r") as fin:
            for line in fin:
                i = np.searchsorted(portions, np.random.rand())
                files[i].write(line)
        for file in files:
            file.close()

    def link_prediction_split(self, graph_file, files, portions):
        """
        Divide a normal graph into a train split and several test splits for link prediction use.
        Each test split contains half true and half false edges.

        Parameters:
            graph_file (str): graph file
            files (list of str): file names,
                the first file is treated as train file
            portions (list of float): split portions
        """
        assert len(files) == len(portions)
        logger.info("splitting graph %s into %s" %
                    (self.relpath(graph_file), ", ".join([self.relpath(file) for file in files])))
        np.random.seed(1024)

        nodes = set()
        edges = set()
        portions = np.cumsum(portions, dtype=np.float32) / np.sum(portions)
        files = [open(file, "w") for file in files]
        num_edges = [0] * len(files)
        with open(graph_file, "r") as fin:
            for line in fin:
                u, v = line.split()[:2]
                nodes.update([u, v])
                edges.add((u, v))
                i = np.searchsorted(portions, np.random.rand())
                if i == 0:
                    files[i].write(line)
                else:
                    files[i].write("%s\t%s\t1\n" % (u, v))
                num_edges[i] += 1

        nodes = list(nodes)
        for file, num_edge in zip(files[1:], num_edges[1:]):
            for _ in range(num_edge):
                valid = False
                while not valid:
                    u = nodes[int(np.random.rand() * len(nodes))]
                    v = nodes[int(np.random.rand() * len(nodes))]
                    valid = u != v and (u, v) not in edges and (v, u) not in edges
                file.write("%s\t%s\t0\n" % (u, v))
        for file in files:
            file.close()

    def image_feature_data(self, dataset, model="resnet50", batch_size=128):
        """
        Compute feature vectors for an image dataset using a neural network.

        Parameters:
            dataset (torch.utils.data.Dataset): dataset
            model (str or torch.nn.Module, optional): pretrained model.
                If it is a str, use the last hidden model of that model.
            batch_size (int, optional): batch size
        """
        import torch
        import torchvision
        from torch import nn

        logger.info("computing %s feature" % model)
        if isinstance(model, str):
            full_model = getattr(torchvision.models, model)(pretrained=True)
            model = nn.Sequential(*list(full_model.children())[:-1])
        num_worker = multiprocessing.cpu_count()
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, num_workers=num_worker, shuffle=False)
        model = model.cuda()
        model.eval()

        features = []
        with torch.no_grad():
            for i, (batch_images, batch_labels) in enumerate(data_loader):
                if i % 100 == 0:
                    logger.info("%g%%" % (100.0 * i * batch_size / len(dataset)))
                batch_images = batch_images.cuda()
                batch_features = model(batch_images).view(batch_images.size(0), -1).cpu().numpy()
                features.append(batch_features)
        features = np.concatenate(features)

        return features


class BlogCatalog(Dataset):
    """
    BlogCatalog social network dataset.

    Splits:
        graph, label, train, test

    Train and test splits are used for link prediction purpose.
    """

    def __init__(self):
        super(BlogCatalog, self).__init__(
            "blogcatalog",
            urls={
                "graph": "https://www.dropbox.com/s/cf21ouuzd563cqx/BlogCatalog-dataset.zip?dl=1",
                "label": "https://www.dropbox.com/s/cf21ouuzd563cqx/BlogCatalog-dataset.zip?dl=1",
                "train": [], # depends on `graph`
                "valid": [], # depends on `graph`
                "test": [] # depends on `graph`
            },
            members={
                "graph": "BlogCatalog-dataset/data/edges.csv",
                "label": "BlogCatalog-dataset/data/group-edges.csv"
            }
        )

    def graph_preprocess(self, raw_file, save_file):
        self.csv2txt(raw_file, save_file)

    def label_preprocess(self, raw_file, save_file):
        self.csv2txt(raw_file, save_file)

    def train_preprocess(self, train_file):
        valid_file = train_file[:train_file.rfind("train.txt")] + "valid.txt"
        test_file = train_file[:train_file.rfind("train.txt")] + "test.txt"
        self.link_prediction_split(self.graph, [train_file, valid_file, test_file], portions=[100, 1, 1])

    def valid_preprocess(self, valid_file):
        train_file = valid_file[:valid_file.rfind("valid.txt")] + "train.txt"
        test_file = valid_file[:valid_file.rfind("valid.txt")] + "test.txt"
        self.link_prediction_split(self.graph, [train_file, valid_file, test_file], portions=[100, 1, 1])

    def test_preprocess(self, test_file):
        train_file = test_file[:test_file.rfind("test.txt")] + "train.txt"
        valid_file = test_file[:test_file.rfind("test.txt")] + "valid.txt"
        self.link_prediction_split(self.graph, [train_file, valid_file, test_file], portions=[100, 1, 1])


class Youtube(Dataset):
    """
    Youtube social network dataset.

    Splits:
        graph, label
    """
    def __init__(self):
        super(Youtube, self).__init__(
            "youtube",
            urls={
                "graph": "http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz",
                "label": "http://socialnetworks.mpi-sws.mpg.de/data/youtube-groupmemberships.txt.gz"
            }
        )

    def label_preprocess(self, raw_file, save_file):
        self.top_k_label(raw_file, save_file, k=47)


class Flickr(Dataset):
    """
    Flickr social network dataset.

    Splits:
        graph, label
    """
    def __init__(self):
        super(Flickr, self).__init__(
            "flickr",
            urls={
                "graph": "http://socialnetworks.mpi-sws.mpg.de/data/flickr-links.txt.gz",
                "label": "http://socialnetworks.mpi-sws.mpg.de/data/flickr-groupmemberships.txt.gz"
            }
        )

    def label_preprocess(self, label_file, save_file):
        self.top_k_label(label_file, save_file, k=5)


class Hyperlink2012(Dataset):
    """
    Hyperlink 2012 graph dataset.

    Splits:
        pld_train, pld_test
    """
    def __init__(self):
        super(Hyperlink2012, self).__init__(
            "hyperlink2012",
            urls={
                "pld_train": "http://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/pld-arc.gz",
                "pld_valid": "http://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/pld-arc.gz",
                "pld_test": "http://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/pld-arc.gz"
            }
        )

    def pld_train_preprocess(self, graph_file, train_file):
        valid_file = train_file[:train_file.rfind("pld_train.txt")] + "pld_valid.txt"
        test_file = train_file[:train_file.rfind("pld_train.txt")] + "pld_test.txt"
        self.link_prediction_split(graph_file, [train_file, valid_file, test_file], portions=[10000, 1, 1])

    def pld_valid_preprocess(self, graph_file, valid_file):
        train_file = valid_file[:valid_file.rfind("pld_valid.txt")] + "pld_train.txt"
        test_file = valid_file[:valid_file.rfind("pld_valid.txt")] + "pld_test.txt"
        self.link_prediction_split(graph_file, [train_file, valid_file, test_file], portions=[10000, 1, 1])

    def pld_test_preprocess(self, graph_file, test_file):
        train_file = test_file[:test_file.rfind("pld_test.txt")] + "pld_train.txt"
        valid_file = test_file[:test_file.rfind("pld_test.txt")] + "pld_valid.txt"
        self.link_prediction_split(graph_file, [train_file, valid_file, test_file], portions=[10000, 1, 1])


class Friendster(Dataset):
    """
    Friendster social network dataset.

    Splits:
        graph, small_graph, label
    """
    def __init__(self):
        super(Friendster, self).__init__(
            "friendster",
            urls={
                "graph": "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz",
                "small_graph": ["https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz",
                                "https://snap.stanford.edu/data/bigdata/communities/com-friendster.all.cmty.txt.gz"],
                "label": "https://snap.stanford.edu/data/bigdata/communities/com-friendster.top5000.cmty.txt.gz"
            }
        )

    def small_graph_preprocess(self, graph_file, label_file, save_file):
        self.induced_graph(graph_file, label_file, save_file)

    def label_preprocess(self, label_file, save_file):
        self.top_k_label(label_file, save_file, k=100, format="(label)-nodes")


class Wikipedia(Dataset):
    """
    Wikipedia dump for word embedding.

    Splits:
        graph
    """
    def __init__(self):
        super(Wikipedia, self).__init__(
            "wikipedia",
            urls={
                "graph": "https://www.dropbox.com/s/q6w950e5f7g7ax8/enwiki-latest-pages-articles-sentences.txt.gz?dl=1"
            }
        )


class Math(Dataset):
    """
    Synthetic math knowledge graph dataset.

    Splits:
        train, valid, test
    """

    NUM_ENTITY = 1000
    NUM_RELATION = 30
    OPERATORS = [
        ("+", lambda x, y: (x + y) % Math.NUM_ENTITY),
        ("-", lambda x, y: (x - y) % Math.NUM_ENTITY),
        ("*", lambda x, y: (x * y) % Math.NUM_ENTITY),
        ("/", lambda x, y: x // y),
        ("%", lambda x, y: x % y)
    ]

    def __init__(self):
        super(Math, self).__init__(
            "math",
            urls={
                "train": [],
                "valid": [],
                "test": []
            }
        )

    def train_preprocess(self, save_file):
        np.random.seed(1023)
        self.generate_math(save_file, num_triplet=20000)

    def valid_preprocess(self, save_file):
        np.random.seed(1024)
        self.generate_math(save_file, num_triplet=1000)

    def test_preprocess(self, save_file):
        np.random.seed(1025)
        self.generate_math(save_file, num_triplet=1000)

    def generate_math(self, save_file, num_triplet):
        with open(save_file, "w") as fout:
            for _ in range(num_triplet):
                i = int(np.random.rand() * len(self.OPERATORS))
                op, f = self.OPERATORS[i]
                x = int(np.random.rand() * self.NUM_ENTITY)
                y = int(np.random.rand() * self.NUM_RELATION) + 1
                fout.write("%d\t%s%d\t%d\n" % (x, op, y, f(x, y)))


class FB15k(Dataset):
    """
    FB15k knowledge graph dataset.

    Splits:
        train, valid, test
    """
    def __init__(self):
        super(FB15k, self).__init__(
            "fb15k",
            urls={
                "train": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k/train.txt",
                "valid": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k/valid.txt",
                "test": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k/test.txt"
            }
        )


class FB15k237(Dataset):
    """
    FB15k-237 knowledge graph dataset.

    Splits:
        train, valid, test
    """
    def __init__(self):
        super(FB15k237, self).__init__(
            "fb15k-237",
            urls={
                "train": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/train.txt",
                "valid": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/valid.txt",
                "test": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/test.txt"
            }
        )


class WN18(Dataset):
    """
    WN18 knowledge graph dataset.

    Splits:
        train, valid, test
    """
    def __init__(self):
        super(WN18, self).__init__(
            "wn18",
            urls={
                "train": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/train.txt",
                "valid": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/valid.txt",
                "test": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/test.txt"
            }
        )


class WN18RR(Dataset):
    """
    WN18RR knowledge graph dataset.

    Splits:
        train, valid, test
    """
    def __init__(self):
        super(WN18RR, self).__init__(
            "wn18rr",
            urls={
                "train": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/train.txt",
                "valid": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/valid.txt",
                "test": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/test.txt"
            }
        )


class Wikidata5m(Dataset):
    """
    Wikidata5m knowledge graph dataset.

    Splits:
        train, valid, test
    """
    def __init__(self):
        super(Wikidata5m, self).__init__(
            "wikidata5m",
            urls={
                "train": "https://www.dropbox.com/s/dty6ufe1gg6keuc/wikidata5m.txt.gz?dl=1",
                "valid": "https://www.dropbox.com/s/dty6ufe1gg6keuc/wikidata5m.txt.gz?dl=1",
                "test": "https://www.dropbox.com/s/dty6ufe1gg6keuc/wikidata5m.txt.gz?dl=1",
                "entity": "https://www.dropbox.com/s/bgmgvk8brjwpc9w/entity.txt.gz?dl=1",
                "relation": "https://www.dropbox.com/s/37jxki93gguv0pp/relation.txt.gz?dl=1",
                "alias2entity": [], # depends on `entity`
                "alias2relation": [] # depends on `relation`
            }
        )

    def train_preprocess(self, graph_file, train_file):
        valid_file = train_file[:train_file.rfind("train.txt")] + "valid.txt"
        test_file = train_file[:train_file.rfind("train.txt")] + "test.txt"
        self.edge_split(graph_file, [train_file, valid_file, test_file], portions=[4000, 1, 1])

    def valid_preprocess(self, graph_file, valid_file):
        train_file = valid_file[:valid_file.rfind("valid.txt")] + "train.txt"
        test_file = valid_file[:valid_file.rfind("valid.txt")] + "test.txt"
        self.edge_split(graph_file, [train_file, valid_file, test_file], portions=[4000, 1, 1])

    def test_preprocess(self, graph_file, test_file):
        train_file = test_file[:test_file.rfind("valid.txt")] + "train.txt"
        valid_file = test_file[:test_file.rfind("train.txt")] + "valid.txt"
        self.edge_split(graph_file, [train_file, valid_file, test_file], portions=[4000, 1, 1])

    def load_alias(self, alias_file):
        alias2object = {}
        ambiguous = set()
        with open(alias_file, "r") as fin:
            for line in fin:
                tokens = line.strip().split("\t")
                object = tokens[0]
                for alias in tokens[1:]:
                    if alias in alias2object and alias2object[alias] != object:
                        ambiguous.add(alias)
                    alias2object[alias] = object
            for alias in ambiguous:
                alias2object.pop(alias)
        return alias2object

    def alias2entity_preprocess(self, save_file):
        return self.load_alias(self.entity)

    def alias2relation_preprocess(self, save_file):
        return self.load_alias(self.relation)


class Freebase(Dataset):
    """
    Freebase knowledge graph dataset.

    Splits:
        train
    """
    def __init__(self):
        super(Freebase, self).__init__(
            "freebase",
            urls={
                "train": "http://commondatastorage.googleapis.com/freebase-public/rdf/freebase-rdf-latest.gz"
            }
        )


class MNIST(Dataset):
    """
    MNIST dataset for visualization.

    Splits:
        train_image_data, train_label_data, test_image_data, test_label_data, image_data, label_data
    """
    def __init__(self):
        super(MNIST, self).__init__(
            "mnist",
            urls={
                "train_image_data": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                "train_label_data": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                "test_image_data": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "test_label_data": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                "image_data": [], # depends on `train_image_data` & `test_image_data`
                "label_data": [] # depends on `train_label_data` & `test_label_data`
            }
        )

    def train_image_data_preprocess(self, raw_file, save_file):
        images = np.fromfile(raw_file, dtype=np.uint8)
        return images[16:].reshape(-1, 28*28)

    def train_label_data_preprocess(self, raw_file, save_file):
        labels = np.fromfile(raw_file, dtype=np.uint8)
        return labels[8:]

    test_image_data_preprocess = train_image_data_preprocess
    test_label_data_preprocess = train_label_data_preprocess

    def image_data_preprocess(self, save_file):
        return np.concatenate([self.train_image_data, self.test_image_data])

    def label_data_preprocess(self, save_file):
        return np.concatenate([self.train_label_data, self.test_label_data])


class CIFAR10(Dataset):
    """
    CIFAR10 dataset for visualization.

    Splits:
        train_image_data, train_label_data, test_image_data, test_label_data, image_data, label_data
    """
    def __init__(self):
        super(CIFAR10, self).__init__(
            "cifar10",
            urls={
                "train_image_data": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                "train_label_data": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                "test_image_data": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                "test_label_data": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                "image_data": [], # depends on `train_image_data` & `test_image_data`
                "label_data": [] # depends on `train_label_data` & `test_label_data`
            },
        )

    def load_images(self, *batch_files):
        images = []
        for batch_file in batch_files:
            batch = np.fromfile(batch_file, dtype=np.uint8)
            batch = batch.reshape(-1, 32*32*3 + 1)
            images.append(batch[:, 1:])
        return np.concatenate(images)

    def load_labels(self, meta_file, *batch_files):
        classes = []
        with open(meta_file, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    classes.append(line)
        classes = np.asarray(classes)
        labels = []
        for batch_file in batch_files:
            batch = np.fromfile(batch_file, dtype=np.uint8)
            batch = batch.reshape(-1, 32*32*3 + 1)
            labels.append(batch[:, 0])
        return classes[np.concatenate(labels)]

    def train_image_data_preprocess(self, raw_path, save_file):
        batch_files = glob.glob(os.path.join(raw_path, "cifar-10-batches-bin/data_batch_*.bin"))
        return self.load_images(*batch_files)

    def train_label_data_preprocess(self, raw_path, save_file):
        meta_file = os.path.join(raw_path, "cifar-10-batches-bin/batches.meta.txt")
        batch_files = glob.glob(os.path.join(raw_path, "cifar-10-batches-bin/data_batch_*.bin"))
        return self.load_labels(meta_file, *batch_files)

    def test_image_data_preprocess(self, raw_path, save_file):
        batch_file = os.path.join(raw_path, "cifar-10-batches-bin/test_batch.bin")
        return self.load_images(batch_file)

    def test_label_data_preprocess(self, raw_path, save_file):
        meta_file = os.path.join(raw_path, "cifar-10-batches-bin/batches.meta.txt")
        batch_file = os.path.join(raw_path, "cifar-10-batches-bin/test_batch.bin")
        return self.load_labels(meta_file, batch_file)

    def image_data_preprocess(self, save_file):
        return np.concatenate([self.train_image_data, self.test_image_data])

    def label_data_preprocess(self, save_file):
        return np.concatenate([self.train_label_data, self.test_label_data])


class ImageNet(Dataset):
    """
    ImageNet dataset for visualization.

    Splits:
        train_image, train_feature_data, train_label, train_hierarchical_label,
        valid_image, valid_feature_data, valid_label, valid_hierarchical_label
    """

    def __init__(self):
        super(ImageNet, self).__init__(
            "imagenet",
            urls={
                "train_image": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar",
                "train_feature_data": [], # depends on `train_image`
                "train_label": [], # depends on `train_image`
                "train_hierarchical_label": [], # depends on `train_image`
                "valid_image": ["http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar",
                                "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz"],
                "valid_feature_data": [], # depends on `valid_image`
                "valid_label": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz",
                "valid_hierarchical_label":
                    "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz",
                "feature_data": [], # depends on `train_feature_data` & `valid_feature_data`
                "label": [], # depends on `train_label` & `valid_label`
                "hierarchical_label": [], # depends on `train_hierarchical_label` & `valid_hierarchical_label`
            }
        )

    def import_wordnet(self):
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")
        from nltk.corpus import wordnet
        try:
            wordnet.synset_from_pos_and_offset
        except AttributeError:
            wordnet.synset_from_pos_and_offset = wordnet._synset_from_pos_and_offset
        return wordnet

    def get_name(self, synset):
        name = synset.name()
        return name[:name.find(".")]

    def readable_label(self, labels, save_file, hierarchy=False):
        wordnet = self.import_wordnet()

        if hierarchy:
            logger.info("generating human-readable hierarchical labels")
        else:
            logger.info("generating human-readable labels")
        synsets = []
        for label in labels:
            pos = label[0]
            offset = int(label[1:])
            synset = wordnet.synset_from_pos_and_offset(pos, offset)
            synsets.append(synset)
        depth = max([synset.max_depth() for synset in synsets])

        num_sample = len(synsets)
        labels = [self.get_name(synset) for synset in synsets]
        num_class = len(set(labels))
        hierarchies = [labels]
        while hierarchy and num_class > 1:
            depth -= 1
            for i in range(num_sample):
                if synsets[i].max_depth() > depth:
                    # only takes the first recall
                    synsets[i] = synsets[i].hypernyms()[0]
            labels = [self.get_name(synset) for synset in synsets]
            hierarchies.append(labels)
            num_class = len(set(labels))
        hierarchies = hierarchies[::-1]

        with open(save_file, "w") as fout:
            for hierarchy in zip(*hierarchies):
                fout.write("%s\n" % "\t".join(hierarchy))

    def image_feature_data(self, image_path):
        """"""
        import torchvision
        from torchvision import transforms

        augmentation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(image_path, augmentation)
        features = super(self, ImageNet).image_feature_data(dataset)

        return features

    def train_image_preprocess(self, image_path, save_file):
        tar_files = glob.glob(os.path.join(image_path, "*.tar"))
        if len(tar_files) == 0:
            return image_path

        for tar_file in tar_files:
            self.extract(tar_file)
            os.remove(tar_file)

        return image_path

    def train_feature_data_preprocess(self, save_file):
        numpy_file = os.path.splitext(save_file)[0] + ".npy"
        if os.path.exists(numpy_file):
            return np.load(numpy_file)
        features = self.image_feature_data(self.train_image)
        np.save(numpy_file, features)
        return features

    def train_label_preprocess(self, save_file):
        image_files = glob.glob(os.path.join(self.train_image, "*/*.JPEG"))
        labels = [os.path.basename(os.path.dirname(image_file)) for image_file in image_files]
        # be consistent with the order in torch.utils.data.DataLoader
        labels = sorted(labels)
        self.readable_label(labels, save_file)

    def train_hierarchical_label_preprocess(self, save_file):
        image_files = glob.glob(os.path.join(self.train_image, "*/*.JPEG"))
        labels = [os.path.basename(os.path.dirname(image_file)) for image_file in image_files]
        # be consistent with the order in torch.utils.data.DataLoader
        labels = sorted(labels)
        self.readable_label(labels, save_file, hierarchy=True)

    def valid_image_preprocess(self, image_path, meta_path, save_file):
        from scipy.io import loadmat

        image_files = glob.glob(os.path.join(image_path, "*.JPEG"))
        if len(image_files) == 0:
            return image_path

        logger.info("re-arranging images into sub-folders")

        image_files = sorted(image_files)
        meta_file = os.path.join(meta_path, "ILSVRC2012_devkit_t12/data/meta.mat")
        id_file = os.path.join(meta_path, "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
        metas = loadmat(meta_file, squeeze_me=True)["synsets"][:1000]
        id2class = {meta[0]: meta[1] for meta in metas}
        ids = np.loadtxt(id_file)
        labels = [id2class[id] for id in ids]
        for image_file, label in zip(image_files, labels):
            class_path = os.path.join(image_path, label)
            if not os.path.exists(class_path):
                os.mkdir(class_path)
            shutil.move(image_file, class_path)

        return image_path

    def valid_feature_data_preprocess(self, save_file):
        numpy_file = os.path.splitext(save_file)[0] + ".npy"
        if os.path.exists(numpy_file):
            return np.load(numpy_file)
        features = self.image_feature_data(self.valid_image)
        np.save(numpy_file, features)
        return features

    def valid_label_preprocess(self, meta_path, save_file):
        from scipy.io import loadmat

        meta_file = os.path.join(meta_path, "ILSVRC2012_devkit_t12/data/meta.mat")
        id_file = os.path.join(meta_path, "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")

        metas = loadmat(meta_file, squeeze_me=True)["synsets"][:1000]
        id2class = {meta[0]: meta[1] for meta in metas}
        ids = np.loadtxt(id_file, dtype=np.int32)
        labels = [id2class[id] for id in ids]
        # be consistent with the order in torch.utils.data.DataLoader
        labels = sorted(labels)
        self.readable_label(labels, save_file)

    def valid_hierarchical_label_preprocess(self, meta_path, save_file):
        from scipy.io import loadmat

        meta_file = os.path.join(meta_path, "ILSVRC2012_devkit_t12/data/meta.mat")
        id_file = os.path.join(meta_path, "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")

        metas = loadmat(meta_file, squeeze_me=True)["synsets"][:1000]
        id2class = {meta[0]: meta[1] for meta in metas}
        ids = np.loadtxt(id_file, dtype=np.int32)
        labels = [id2class[id] for id in ids]
        # be consistent with the order in torch.utils.data.DataLoader
        labels = sorted(labels)
        self.readable_label(labels, save_file, hierarchy=True)

    def feature_data_preprocess(self, save_file):
        return np.concatenate([self.train_feature_data, self.valid_feature_data])

    def label_preprocess(self, save_file):
        with open(save_file, "w") as fout:
            with open(self.train_label, "r") as fin:
                shutil.copyfileobj(fin, fout)
        with open(save_file, "a") as fout:
            with open(self.valid_label, "r") as fin:
                shutil.copyfileobj(fin, fout)

    def hierarchical_label_preprocess(self, save_file):
        with open(save_file, "w") as fout:
            with open(self.train_hierarchical_label, "r") as fin:
                shutil.copyfileobj(fin, fout)
            with open(self.valid_hierarchical_label, "r") as fin:
                shutil.copyfileobj(fin, fout)


blogcatalog = BlogCatalog()
youtube = Youtube()
flickr = Flickr()
hyperlink2012 = Hyperlink2012()
friendster = Friendster()
wikipedia = Wikipedia()

math = Math()
fb15k = FB15k()
fb15k237 = FB15k237()
wn18 = WN18()
wn18rr = WN18RR()
wikidata5m = Wikidata5m()
freebase = Freebase()

mnist = MNIST()
cifar10 = CIFAR10()
imagenet = ImageNet()

__all__ = [
    "Dataset",
    "BlogCatalog", "Youtube", "Flickr", "Hyperlink2012", "Friendster", "Wikipedia",
    "Math", "FB15k", "FB15k237", "WN18", "WN18RR", "Wikidata5m", "Freebase",
    "MNIST", "CIFAR10", "ImageNet"
]