Customize Solvers
=================

A more interesting thing to explore is extending GraphVite with new solvers.
Generally, the core library is capable to perform any graph embedding variant that
fits into the following paradigm.

- The training samples are edges.
  There may be additional attributes (e.g. labels) to edge samples.

To support that, GraphVite provides a protocol interface and a series of abstract
classes. You only need to declare the protocols for your parameters, and fill in the
virtual member functions for the classes.

Let's begin with the protocol interface. There are 3 main protocols for parameters.

- ``head``
- ``tail``
- ``global``

For each parameter matrix, it should be assigned one of these protocols.
``head`` means that the parameter matrix is indexed by head nodes in directed edges,
while ``tail`` corresponds to tail nodes. Any other parameter matrix should be assigned
with ``global``.

There are also 2 optional protocols. One is ``in place``, which implies that the
parameter matrix takes in-place update and doesn't need storage for gradients.
The other is ``shared``, which implies the matrix is shared with the previous one.
This may be used for tied weight case.

Each parameter matrix should also be specified with a shape. You can use ``auto``
if the shape can be inferred from the protocol and the graph structure.

For example, knowledge graph embeddings take the following settings.

.. code-block:: c++

    // head embeddings, tail embeddings, relation embeddings
    protocols = {head | in place, tail | in place | shared, global};
    shapes = {auto, auto, graph->num_relation};

If your learning routine also needs negative sampling, you should additionally
specify a negative sampler protocol. For knowledge graph embedding, this is

.. code-block:: c++

    negative_sampler_protocol = head | tail;

Given the protocols, GraphVite will automatically schedule the paramters and samples
over multiple GPUs, using an algorithm called parallel negative sampling. For a more
detailed explanation of the algorithm, see section 3.2 in `GraphVite paper`_.

.. _GraphVite paper: https://arxiv.org/pdf/1903.00757.pdf

.. note::
    Parallel negative sampling only takes place when at least one parameter matrix
    is ``head`` or ``tail``. If all parameters are ``global``, GraphVite will schedule
    them by standard data parallel.

To implement a new solver, you need to implement ``get_protocols()``,
``get_sampler_protocol()`` and ``get_shapes()`` as above. Some additional helper
functions may be required to complete the solver.

A solver also contains a sampler and a worker class. By default, the sampler samples
positive edges from the graph, with probability proportional to the weight of each
edge. You only need to specify the additional edge attributes in ``get_attributes()``.

For the worker, it will build the negative sampler according to the its protocol.
You need to specify the GPU implementation of models in ``kernel_dispatch()``. See
:doc:`model` for how to do that.

Finally, to get your new solver appeared in Python, add a Python declaration for it in
``include/bind.h``, and instantiate it in ``src/graphvite.cu``.

See ``include/instance/*`` for all solver instances.

.. note::
    Functions in solver, sampler and worker can be overrided. For example,
    :class:`GraphSolver <graphvite.solver.GraphSolver>` overrides edge sampling with
    online augmentation.