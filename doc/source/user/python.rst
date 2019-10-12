Python Interface
================

GraphVite provides Python interface for convenient integration with other software.
To use GraphVite in Python, import these two modules in our script.

    >>> import graphvite as gv
    >>> import graphvite.application as gap

The ``graphvite`` module itself provides basic class interface, such as graphs,
solvers, optimizers and datasets. The ``application`` module contains high-level
wrappers of applications, along with their evaluation routines.

Applications
------------

We can invoke a node embedding application with the following lines.

    >>> app = gap.GraphApplication(dim=128)
    >>> app.load(file_name=gv.dataset.blogcatalog.train)
    >>> app.build()
    >>> app.train()
    >>> app.evaluate("node classification", file_name=gv.dataset.blogcatalog.label)

where the arguments of each member function are identical to those in the
:doc:`configuration files <configuration>`.

.. seealso::
    Package reference: :doc:`Application <../api/application>`

Basic classes
-------------

The basic classes are very helpful if we need fine-grained manipulation of the
pipeline. For example, we may train an ensemble of node embedding models on the
same graph. First, create a graph and two node embedding solvers.

    >>> graph = gv.graph.Graph()
    >>> graph.load(gv.dataset.blogcatalog.train)
    >>> solvers = [gv.solver.GraphSolver(dim=128, device_ids=[gpu], num_sampler_per_worker=4)
    ...            for gpu in range(2)]

Then, build the solvers on that graph. This step determines all memory allocation.

    >>> for solver in solvers:
    >>>     solver.build(graph)

Now we can train the solver. The training stage of solvers can be fully paralleled
with multiple threads, since GraphVite never holds Python GIL inside basic classes.

    >>> from multiprocessing.pool import ThreadPool
    >>> pool = ThreadPool(2)
    >>> models = ["DeepWalk", "LINE"]
    >>> pool.map(lambda x: x[0].train(x[1]), zip(solvers, models))

Finally, obtain the ensembled embeddings.

    >>> import numpy as np
    >>> vertex_embeddings = np.hstack([s.vertex_embeddings for s in solvers])
    >>> context_embeddings = np.hstack([s.context_embeddings for s in solvers])

Note the embeddings are stored in an internal order. To get an index of a specific
node, use the ``name2id`` property of the graph. For example, the following line
prints the vertex embedding of node "1024".

    >>> print(vertex_embeddings[graph.name2id["1024"]])

.. seealso::
    Package reference: :doc:`Graph <../api/graph>`, :doc:`Solver <../api/solver>`,
    :doc:`Optimizer <../api/optimizer>`, :doc:`Dataset <../api/dataset>`

Logging settings
----------------

GraphVite outputs a bunch of messages during stages like training. We can set the
logging level to dismiss unimportant logs.

The following lines suppress most logs except hyperparameters and evaluation results.
The verbose mode additionally prints time tags and thread IDs each log.

    >>> import logging
    >>> gv.init_logging(logging.WARNING, verbose=True)

Messages can be also redirected to files by specifying a value for the ``dir``
argument.