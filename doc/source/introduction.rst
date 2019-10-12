GraphVite - graph embedding at high speed and large scale
=========================================================

.. include:: link.rst

GraphVite is a general graph embedding engine, dedicated to high-speed and
large-scale embedding learning in various applications. By cooperating CPUs and GPUs
for learning, it scales to million-scale or even billion-scale graphs. With its
Python interface, you can easily practice advanced graph embedding algorithms, and
get results in incredibly short time.

Try GraphVite if you have any of the following demands.

- You want to reproduce graph learning algorithms on a uniform platform.
- You need fast visualization for graphs or high-dimensional data.
- You are tired of waiting a long time for prototyping or tuning models.
- You need to learn representations of large graphs or knowledge graphs.

Generally, GraphVite provides complete training and evaluation pipelines for 3
applications: **node embedding**, **knowledge graph embedding** and
**graph & high-dimensional data visualization**. Besides, it also includes 9 popular
models, along with their benchmarks on a bunch of standard datasets.

.. figure:: ../../asset/graph.png
    :align: left
    :height: 180px
    :target: overview.html#node-embedding
    :figclass: align-center

    Node Embedding

.. figure:: ../../asset/knowledge_graph.png
    :align: left
    :height: 180px
    :target: overview.html#knowledge-graph-embedding
    :figclass: align-center

    Knowledge Graph |br| Embedding

.. figure:: ../../asset/visualization.png
    :align: left
    :height: 180px
    :target: overview.html#graph-high-dimensional-data-visualization
    :figclass: align-center

    Graph & |br| High-dimensional |br| Data Visualization

.. |br| raw:: html

    <br>

.. raw:: html

    <div style="clear: both" />

How fast is GraphVite?
----------------------

To give a brief idea of GraphVite's speed, we summarize the training time of
GraphVite along with the best open-source implementations. All the time is reported
based on a server with 24 CPU threads and 4 V100 GPUs.

Training time of node embedding on `Youtube`_ dataset.

+-------------+----------------------------+-----------+---------+
| Model       | Existing Implementation    | GraphVite | Speedup |
+=============+============================+===========+=========+
| `DeepWalk`_ | `1.64 hrs (CPU parallel)`_ | 1.19 mins | 82.9x   |
+-------------+----------------------------+-----------+---------+
| `LINE`_     | `1.39 hrs (CPU parallel)`_ | 1.17 mins | 71.4x   |
+-------------+----------------------------+-----------+---------+
| `node2vec`_ | `24.4 hrs (CPU parallel)`_ | 4.39 mins | 334x    |
+-------------+----------------------------+-----------+---------+

.. _1.64 hrs (CPU parallel): https://github.com/phanein/deepwalk
.. _1.39 hrs (CPU parallel): https://github.com/tangjianpku/LINE
.. _24.4 hrs (CPU parallel): https://github.com/aditya-grover/node2vec

Training / evaluation time of knowledge graph embedding on `FB15k`_ dataset.

+-----------+---------------------------------+--------------------+---------------+
| Model     | Existing Implementation         | GraphVite          | Speedup       |
+===========+=================================+====================+===============+
| `TransE`_ | `1.31 hrs / 1.75 mins (1 GPU)`_ | 13.5 mins / 54.3 s | 5.82x / 1.93x |
+-----------+---------------------------------+--------------------+---------------+
| `RotatE`_ | `3.69 hrs / 4.19 mins (1 GPU)`_ | 28.1 mins / 55.8 s | 7.88x / 4.50x |
+-----------+---------------------------------+--------------------+---------------+

.. _1.31 hrs / 1.75 mins (1 GPU): https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
.. _3.69 hrs / 4.19 mins (1 GPU): https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

Training time of high-dimensional data visualization on `MNIST`_ dataset.

+-------------+-----------------------------+-----------+---------+
| Model       | Existing Implementation     | GraphVite | Speedup |
+=============+=============================+===========+=========+
| `LargeVis`_ | `15.3 mins (CPU parallel)`_ | 13.9 s    | 66.8x   |
+-------------+-----------------------------+-----------+---------+

.. _15.3 mins (CPU parallel): https://github.com/lferry007/LargeVis

Comparison to concurrent work
-----------------------------

A work concurrent to GraphVite is `PyTorch-BigGraph`_, which aims at accelerating
knowledge graph embedding on large-scale data. Here is an apple-to-apple comparison
of models implemented in both libraries on `FB15k`_, under the same setting of
hyperparameters.

.. _PyTorch-BigGraph: https://torchbiggraph.readthedocs.io

+-------------+------------------+-----------+---------+
| Model       | PyTorch-BigGraph | GraphVite | Speedup |
+=============+==================+===========+=========+
| `TransE`_   | 1.21 hrs         | 8.37 mins | 8.70x   |
+-------------+------------------+-----------+---------+
| `DistMult`_ | 2.48 hrs         | 20.3 mins | 7.33x   |
+-------------+------------------+-----------+---------+
| `ComplEx`_  | 3.13 hrs         | 18.5 mins | 10.1x   |
+-------------+------------------+-----------+---------+

GraphVite surpasses its counterpart by a signficant margin. Besides, the framework of
GraphVite also supports two more applications, and provides many benchmarks for easy
research and development.

About the name
--------------
GraphVite(/ɡɹæfvit/) is a combination of English word "graph" and French word
"vite", which means "rapid". GraphVite represents the traits of this library,
as well as the bilingual environment of `Mila`_ where the library was developed.

.. _Mila: https://mila.quebec