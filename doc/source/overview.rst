Application Overview
====================

In GraphVite, the pipelines are packaged into classes, which we call applications.

There are 3 main applications, node embedding, knowledge graph embedding, and
graph & high-dimensional data visualization. For each application, GraphVite loads
an input graph, perfoms embedding training, and finally evaluates the embeddngs on
downstream tasks.

.. _node embedding:

Node Embedding
--------------

Node embedding is a family of algorithms that learn a representation for each node
in a graph. It is important for graph analysis and a variety of downstream tasks.

For example, node embedding can be leveraged for analyzing social networks, citation
networks, or protein-protein interaction networks. It may be also helpful to other
unsupervised learning problems with graph structures.

To qualify the learned embeddings, we evaluate them on the node classification and
link prediction tasks.

.. seealso::
    Package Reference:
    :class:`GraphApplication <graphvite.application.GraphApplication>`

.. _knowledge graph embedding:

Knowledge Graph Embedding
-------------------------

Knowledge graph (aka. knowledge base) is a family of graphs where each edge has a
type, indicating the relation of the connected nodes. In knowledge graphs, nodes
are called entities, and edges are called relations. The knowledge graph embedding
algorithm aims to learn a representation for each entity and relation.

With knowledge graph embeddings, it is easy to compare entities or relations in a
uniform space, and further infer unobserved links in a knowledge graph.

The learned embeddings are evaluated under the link prediction task in GraphVite.

.. seealso::
    Package Reference:
    :class:`KnowledgeGraphApplication <graphvite.application.KnowledgeGraphApplication>`

.. _visualization:

Graph & High-dimensional Data Visualization
-------------------------------------------

Visualization is a critical step in exploring and analyzing graphs and
high-dimensional data. Typically, visualization methods project each data points into
a low-dimensional space.

As most projection methods treat the similarity between data points as a graph,
GraphVite is also able to provide acceleration for this application. Taking a graph
or a group of high-dimensional vectors, GraphVite can produce either 2D or 3D
projections in a very short time.

.. seealso::
    Package Reference:
    :class:`VisualizationApplication <graphvite.application.VisualizationApplication>`