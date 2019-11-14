Data Format
===========

GraphVite is designed to support a wide range of formats for graphs. Generally, it
doesn't enforce any type restriction on input elements. We can either use integers or
strings as our input. Each line in the file is parsed as

.. code-block::

    [token] [delimiter] [token] [delimiter]... [comment]...

By default, GraphVite treats any blank character as delimiter, and string after ``#``
as comment. You can change these settings in the
:ref:`format section <experiment configuration>` of configuration files, or using
``app.set_format(delimiters, comment)`` in Python code.

GraphVite can also construct graphs from Python objects, which is helpful if graphs
are dynamically generated. It takes a nested list similar to the file format. Each
token should be a string or a float.

.. code-block:: python

    graph = [[token, token], [token, token], ...]

Node Embedding
--------------

The input graph for node embedding follows the edge list format. Each line should be

.. code-block::

    [head] [tail]

You may also specify a weight for each edge.

.. code-block::

    [head] [tail] [weight]

For link prediction task, the evaluation file consists of edges and labels.

.. code-block::

    [head] [tail] [label]

where label ``1`` is positive and ``0`` is negative. The filter file takes the same
format as the input graph.

For node classification task, each line is a node and a label. If a node has more
than one label, it should take multiple lines.

.. code-block::

    [node] [label]

Knowledge Graph Embedding
-------------------------

Each line in a knowledge graph is a triplet.

.. code-block::

    [head] [relation] [tail]

You may also specify a weight for each triplet.

.. code-block::

    [head] [relation] [tail] [weight]

All the files in knowledge graph evaluation tasks take the same triplet format.

Graph & High-dimensional Data Visualization
-------------------------------------------

For graph visualization, the input format is same as the graph in node embedding.

For high-dimensional data visualization, the input format can either be a 2D numpy
array or a text matrix. Each row in the matrix is parsed as a point in the
high-dimensional space.