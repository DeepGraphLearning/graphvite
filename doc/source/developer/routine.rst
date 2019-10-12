Customize Routine
=================

For advanced developers, GraphVite also supports customizing routines, such as
training and prediction. Here we will illustrate how to add a new routine to the
knowledge graph solver.

Before we start, it would be better if you know some basics about
`the index and threads`_ in CUDA. In GraphVite, the threads are arranged in a group
of 32 (`warp`_). Threads in a group works simultaneously on an edge sample, where
each thread is responsible for computation in some dimensions, according to the
modulus of the dimension.

.. _the index and threads: https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#Indexing
.. _warp: https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#Warps

First, get into ``include/instance/gpu/knowledge_graph.h``. This file includes several
training functions and a prediction function.

.. code-block:: c++

    template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
    __global__ void train(...)

    template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
    __global__ void train_1_moment(...)

    template<class Vector, class Index, template<class> class Model, OptimizerType optimizer_type>
    __global__ void train_2_moment(...)

    template<class Vector, class Index, template<class> class Model>
    __global__ void predict(...)

The 3 implementations correspond to 3 categories of optimizers, as we have seen in
:doc:`routine`. Routines with different numbers of moment statistics are separated
to achieve maximal compile-time optimization.

Let's take a look at a training function. Generally, the function body looks like

.. code-block:: c++

    for (int sample_id = thread_id / kWarpSize; sample_id < batch_size; sample_id += num_thread / kWarpSize) {
        if (adversarial_temperature > kEpsilon)
            for (int s = 0; s < num_negative; s++)
                normalizer += ...;

        for (int s = 0; s <= num_negative; s++) {
            model.forward(sample[s], logit);
            prob = sigmoid(logit);

            gradient = ...;
            weight = ...;
            sample_loss += ...;
            model.backward<optimizer_type>(sample[s], gradient);
        }
    }

The outer loop iterates over all positive samples. For each positive sample and its
negative samples, we first compute the normalizer of self-adversarial negative
sampling, and then perform forward and backward propagation for each sample.

For example, if we want to change the negative log likelihood to a mean square error,
we can change the following lines.

.. code-block:: c++

    gradient = 2 * (logit - label);
    sample_loss += weight * (logit - label) * (logit - label);

Or we can use a margin-based ranking loss like

.. code-block:: c++

    model.forward(samples[num_negative], positive_score); // the positive sample

    for (int s = 0; s < num_negative; s++) {
        model.forward(samples[s], negative_logit);
        if (positive_score - negative_score < margin) {
            sample_loss += negative_score - positive_score + margin;
            gradient = 1;
            model.backward<optimizer_type>(sample[s], gradient);
            model.backward<optimizer_type>(sample[num_negative], -gradient);
        }
    }

We may also add new hyperparameters or training routines. Note if we change
the signature of the function, we should also update its calls accrodingly. For
knowledge graph, they are in ``train_dispatch()`` and ``predict_dispatch()`` of file
``include/instance/knowledge_graph.cuh``.