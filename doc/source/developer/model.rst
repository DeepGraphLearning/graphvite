Customize Models
================

One common demand for graph embedding is to customize the model (i.e. loss function).
Here we will show you an example of adding a new loss function to the knowledge graph
solver.

Before start, it would be better if you know some basics about `the index and threads`_
in CUDA. In GraphVite, the threads are arranged in a group of 32 (`warp`_). Threads in
a group works simultaneously on an edge sample, where each thread is responsible for
computation in some dimensions, according to the modulus of the dimension.

.. _the index and threads: https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#Indexing
.. _warp: https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#Warps

First, get into ``include/gpu/knowledge_graph.h``. Fork an existing loss function
(e.g. transe) and change it to your own name.

You will find 3 implementations of the loss function in the namespace.

.. code-block:: c++

    namespace transe {
    __global__ void train(...);
    __global__ void train_1_moment(...);
    __global__ void train_2_moment(...);
    }

The three implementations correspond to 3 categories of optimizers. We are going to
modify one, and then do some copy-and-paste work to the others.

Let's start from ``train_2_moment()``. Find the following two loops.
You can locate them by searching ``i = lane_id``.

.. code-block:: c++

    for (int i = lane_id; i < dim; i += kWarpSize) {
        x += ...;
    }

    for (int i = lane_id; i < dim; i += kWarpSize) {
        head[i] -= (optimizer.*update)(head[i], ..., head_moment1[i], head_moment2[i], weight);
        tail[i] -= (optimizer.*update)(tail[i], ..., tail_moment1[i], tail_moment2[i], weight);
        Float relation_update = (optimizer.*update)(relation[i], ...,
                                                    relation_moment1[i], relation_moment2[i], weight);
        relation[i] -= relation_update;
        relation_gradient[i] += relation_update;
    }

The first loop is the forward propagtion, which computes the score for each dimension.
The second loop is the backward propagation, which computes the gradient for each
dimension.

What you need to do is to replace the ellipsis with your own formulas.
Note the head gradient is already stored in ``gradient``, which you need to refer
in your back propagation.

If you want to change the loss function over the logit
(e.g. change from margin loss to standard log-likelihood), you need also change
the code between these two loops, as the following fragment shows.

.. code-block:: c++

    x = WarpBroadcast(WarpReduce(x), 0);
    Float prob = ...;
    if (label) {
        gradient = ...;
        weight = ...;
    #ifdef USE_LOSS
        sample_loss += ...;
    #endif
    } else {
        gradient = ...;
        weight = ...;
    #ifdef USE_LOSS
        sample_loss += ...;
    #endif
    }

Now you are almost there. Copy the modified fragment to ``train()`` and
``train_1_moment()``, and delete undeclared variables like ``head_moment2``.
Now your model supports all optimizers.

Finally, you have to let the solver know there is a new model. In
``instance/knowledge_graph.cuh``, add the name of your model in
``get_available_models()``. Also add run-time dispatch for optimizers in
``kernel_dispatch()``.

.. code-block:: c++

    switch (num_moment) {
        case 0:
            if (solver->model == ...)
                ...
        case 1:
            if (solver->model == ...)
                ...
        case 2:
            if (solver->model == ...)
                ...

Compile the source and it should be ready.