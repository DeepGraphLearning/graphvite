Magic of Auto
=============

Hyperparameter tuning is usually painful for machine learning practioners. In order
to help users focus on the most important part, GraphVite provides an auto deduction
for many hyperparameters. Generally, auto deduction will maximize the speed of the
system, while keep the performance loss as small as possible.

To invoke auto deduction, we can simply leave hyperparameters to their default
values. An explicit way is to use ``auto`` in configuration files, or value
``gv.auto`` in Python.

Here lists hyperparameters that support auto deduction.

.. code-block:: yaml

    resource:
        gpus: []
        gpu_memory_limit: auto
        cpu_per_gpu: auto

    build:
        optimizer: auto
        num_partition: auto
        episode_size: auto

    train:
        # for node embedding
        augmentation_step: auto

.. note::
    The auto value for ``gpus`` is an empty list.