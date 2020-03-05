![GraphVite logo](asset/logo/logo.png)

GraphVite - graph embedding at high speed and large scale
=========================================================

[![Install with conda](https://anaconda.org/milagraph/graphvite/badges/installer/conda.svg)][conda]
[![License](https://anaconda.org/milagraph/graphvite/badges/license.svg)][license]
[![Downloads](https://anaconda.org/milagraph/graphvite/badges/downloads.svg)][conda]

[conda]: https://anaconda.org/milagraph/graphvite
[license]: LICENSE

[Docs] | [Tutorials] | [Benchmarks] | [Pre-trained Models]

[Docs]: https://graphvite.io/docs/latest/api/application
[Tutorials]: https://graphvite.io/tutorials
[Benchmarks]: https://graphvite.io/docs/latest/benchmark
[Pre-trained Models]: https://graphvite.io/docs/latest/pretrained_model

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

Training time of node embedding on [Youtube] dataset.

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

Training / evaluation time of knowledge graph embedding on [FB15k] dataset.

| Model           | Existing Implementation           | GraphVite          | Speedup       |
|-----------------|-----------------------------------|--------------------|---------------|
| [TransE]        | [1.31 hrs / 1.75 mins (1 GPU)][3] | 13.5 mins / 54.3 s | 5.82x / 1.93x |
| [RotatE]        | [3.69 hrs / 4.19 mins (1 GPU)][4] | 28.1 mins / 55.8 s | 7.88x / 4.50x |

[FB15k]: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
[TransE]: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf
[RotatE]: https://arxiv.org/pdf/1902.10197.pdf
[3]: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
[4]: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

Training time of high-dimensional data visualization on [MNIST] dataset.

| Model        | Existing Implementation       | GraphVite | Speedup |
|--------------|-------------------------------|-----------|---------|
| [LargeVis]   | [15.3 mins (CPU parallel)][5] | 13.9 s    | 66.8x   |

[MNIST]: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
[LargeVis]: https://arxiv.org/pdf/1602.00370.pdf
[5]: https://github.com/lferry007/LargeVis

Requirements
------------

Generally, GraphVite works on any Linux distribution with CUDA >= 9.2.

The library is compatible with Python 2.7 and 3.6/3.7.

Installation
------------

### From Conda ###

```bash
conda install -c milagraph -c conda-forge graphvite cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")
```

If you only need embedding training without evaluation, you can use the following
alternative with minimal dependencies.

```bash
conda install -c milagraph -c conda-forge graphvite-mini cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")
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

### On Colab ###

```bash
!wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!./Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local -f

!conda install -y -c milagraph -c conda-forge graphvite \
    python=3.6 cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+\.\d+")
!conda install -y wurlitzer ipykernel
```

```python
import site
site.addsitedir("/usr/local/lib/python3.6/site-packages")
%reload_ext wurlitzer
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
loss = 0.371041

------------- link prediction --------------
AUC: 0.899933

----------- node classification ------------
macro-F1@20%: 0.242114
micro-F1@20%: 0.391342
```

Baseline Benchmark
------------------

To reproduce a baseline benchmark, you only need to specify the keywords of the
experiment. e.g. model and dataset.

```bash
graphvite baseline [keyword ...] [--no-eval] [--gpu n] [--cpu m] [--epoch e]
```

You may also set the number of GPUs and the number of CPUs per GPU.

Use ``graphvite list`` to get a list of available baselines.

Custom Experiment
-----------------

Create a yaml configuration scaffold for graph, knowledge graph, visualization or
word graph.

```bash
graphvite new [application ...] [--file f]
```

Fill some necessary entries in the configuration following the instructions. You
can run the configuration by

```bash
graphvite run [config] [--no-eval] [--gpu n] [--cpu m] [--epoch e]
```

High-dimensional Data Visualization
-----------------------------------

You can visualize your high-dimensional vectors with a simple command line in
GraphVite.

```bash
graphvite visualize [file] [--label label_file] [--save save_file] [--perplexity n] [--3d]
```

The file can be either a numpy dump `*.npy` or a text matrix `*.txt`. For the save
file, we recommend to use `png` format, while `pdf` is also supported.

Contributing
------------

We welcome all contributions from bug fixs to new features. Please let us know if you
have any suggestion to our library.

Development Team
----------------

GraphVite is developed by [MilaGraph], led by Prof. [Jian Tang].

Authors of this project are [Zhaocheng Zhu], [Shizhen Xu], [Meng Qu] and [Jian Tang].
Contributors include [Kunpeng Wang] and [Zhijian Duan].

[MilaGraph]: https://github.com/DeepGraphLearning
[Zhaocheng Zhu]: https://kiddozhu.github.io
[Shizhen Xu]: https://github.com/xsz
[Meng Qu]: https://mnqu.github.io
[Jian Tang]: https://jian-tang.com
[Kunpeng Wang]: https://github.com/Kwinpeng
[Zhijian Duan]: https://github.com/zjduan

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