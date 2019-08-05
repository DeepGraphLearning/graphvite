![GraphVite logo](asset/logo/logo.png)

GraphVite - graph embedding at high speed and large scale
=========================================================

[![Install with conda](https://anaconda.org/milagraph/graphvite/badges/installer/conda.svg)][conda]
[![License](https://anaconda.org/milagraph/graphvite/badges/license.svg)][license]

[conda]: https://anaconda.org/milagraph/graphvite
[license]: LICENSE

GraphVite is a general graph embedding engine, dedicated to high-speed and
large-scale embedding learning in various applications.

GraphVite provides complete training and evaluation pipelines for 3 applications:
**node embedding**, **knowledge graph embedding** and
**graph & high-dimensional data visualization**. Besides, it also includes 9 popular
models, along with their benchmarks on a bunch of standard datasets.

<table align="center" style="text-align:center">
    <tr>
        <th>Node Embedding</th>
        <th>Knowledge Graph Embedding</th>
        <th>Graph & High-dimensional Data Visualization</th>
    </tr>
    <tr>
        <td><img src="asset/graph.png" height="240" /></td>
        <td><img src="asset/knowledge_graph.png" height="240" /></td>
        <td><img src="asset/visualization.png" height="240" /></td>
    </tr>
</table>

Here is a summary of the training time of GraphVite along with the best open-source
implementations on 3 applications. All the time is reported based on a server with
24 CPU threads and 4 V100 GPUs.

Node embedding on [Youtube] dataset.

| Model      | Existing Implementation       | GraphVite | Speedup |
|------------|-------------------------------|-----------|---------|
| [DeepWalk] | [1.64 hrs (CPU parallel)][1]  | 1.19 mins | 82.9x   |
| [LINE]     | [1.39 hrs (CPU parallel)][2]  | 1.17 mins | 71.4x   |
| [node2vec] | [24.4 hrs (CPU parallel)][3]  | 4.39 mins | 334x    |

[Youtube]: http://conferences.sigcomm.org/imc/2007/papers/imc170.pdf
[DeepWalk]: https://arxiv.org/pdf/1403.6652.pdf
[LINE]: https://arxiv.org/pdf/1503.03578.pdf
[node2vec]: https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf
[1]: https://github.com/phanein/deepwalk
[2]: https://github.com/tangjianpku/LINE
[3]: https://github.com/aditya-grover/node2vec

Knowledge graph embedding on [FB15k] dataset.

| Model           | Existing Implementation       | GraphVite | Speedup |
|-----------------|-------------------------------|-----------|---------|
| [TransE]        | [1.31 hrs (1 GPU)][3]         | 14.8 mins | 5.30x   |
| [RotatE]        | [3.69 hrs (1 GPU)][4]         | 27.0 mins | 8.22x   |

[FB15k]: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
[TransE]: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
[RotatE]: https://arxiv.org/pdf/1902.10197.pdf
[3]: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
[4]: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

High-dimensional data visualization on [MNIST] dataset.

| Model        | Existing Implementation       | GraphVite | Speedup |
|--------------|-------------------------------|-----------|---------|
| [LargeVis]   | [15.3 mins (CPU parallel)][5] | 15.1 s    | 60.8x   |

[MNIST]: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
[LargeVis]: https://arxiv.org/pdf/1602.00370.pdf
[5]: https://github.com/lferry007/LargeVis

Requirements
------------

Generally, GraphVite works on any Linux distribution with CUDA >= 9.2.

The library is compatible with Python 2.7 and 3.5/3.6/3.7.

Installation
------------

### From Conda ###

GraphVite can be installed through conda with only one line.

```bash
conda install -c milagraph graphvite
```

If you only need embedding training without evaluation, you can use the following
alternative with minimal dependencies.

```bash
conda install -c milagraph graphvite-mini
```

### From Source ###

Before installation, make sure you have `conda` installed.

```bash
git clone https://github.com/DeepGraphLearning/graphvite
cd graphvite
conda install -y --file conda/requirements.txt
mkdir build
cd build && cmake .. && make && cd -
cd python && python setup.py install && cd -
```

Quick Start
-----------

Here is a quick-start example of the node embedding application.

```bash
graphvite baseline quick start
```

Typically, the example takes no more than 1 minute. You will obtain some output like

```
Batch id: 6000
loss = 0.371641

macro-F1@20%: 0.236794
micro-F1@20%: 0.388110
```

Baseline Benchmark
------------------

To reproduce a baseline benchmark, you only need to specify the keywords of the
experiment. e.g. model and dataset.

```bash
graphvite baseline [keyword ...] [--no-eval] [--gpu n] [--cpu m]
```

You may also set the number of GPUs and the number of CPUs per GPU.

Use ``graphvite list`` to get a list of available baselines.

High-dimensional Data Visualization
-----------------------------------

You can visualize your high-dimensional vectors with a simple command line in
GraphVite.

```bash
graphvite visualize [file] [--label label_file] [--save save_file] [--perplexity n] [--3d]
```

The file can be either in numpy dump or text format. For the save file, we recommend
to use a `png` format, while `pdf` is also supported.

Contributing
------------

We welcome all contributions from bug fixs to new features. Please let us know if you
have any suggestion to our library.

Development Team
----------------

GraphVite is developed by [MilaGraph], led by Prof. Jian Tang.

Authors of this project are Zhaocheng Zhu, Shizhen Xu, Meng Qu and Jian Tang.
Contributors include Kunpeng Wang and Zhijian Duan.

[MilaGraph]: https://github.com/DeepGraphLearning

Citation
--------

If you find GraphVite useful for your research or development, please cite the
following [paper].

[paper]: https://arxiv.org/pdf/1903.00757.pdf

```
@inproceedings{zhu2019graphvite,
    title={GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding},
     author={Zhu, Zhaocheng and Xu, Shizhen and Qu, Meng and Tang, Jian},
     booktitle={The World Wide Web Conference},
     pages={2494--2504},
     year={2019},
     organization={ACM}
 }
```

Acknowledgements
----------------

We would like to thank Compute Canada for supporting GPU servers. We specially thank
Wenbin Hou for useful discussions on C++ and GPU programming techniques.