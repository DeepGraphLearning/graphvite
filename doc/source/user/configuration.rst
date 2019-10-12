Configuration Files
===================

.. include:: ../link.rst

.. _experiment configuration:

Experiment configuration
------------------------

An experiment configuration starts with an ``application type``, and contains settings
for ``resource``, ``format``, ``graph``, ``build``, ``train``, ``evaluate`` and
``save`` stages.

Here is the configuration used in :doc:`../quick_start`.
:download:`quick_start.yaml <../../../config/demo/quick_start.yaml>`

The stages are configured as follows.

.. code-block:: yaml

    application: [type]

The application type can be ``graph``, ``word graph``, ``knowledge graph`` or
``visualization``.

.. code-block:: yaml

    resource:
      gpus: [list of GPU ids]
      gpu_memory_limit: [limit for each GPU in bytes]
      cpu_per_gpu: [CPU thread per GPU]
      dim: [dim]

.. note::
    For optimal performance, modules are compiled with pre-defined dimensions in C++.
    As a drawback, only dimensions that are powers of 2 are supported in the library.

.. code-block:: yaml

    format:
      delimiters: [string of delimiter characters]
      comment: [prefix of comment strings]

Format section is optional. By default, delimiters are any blank character and comment
is "#", following the Python style.

.. code-block:: yaml

    graph:
      file_name: [file name]
      as_undirected: [symmetrize the graph or not]

For standard datasets, we can specify its file name by ``<[dataset].[split]>``.
This would make the configuration file independent of the path.

.. code-block:: yaml

    build:
      optimizer:
        type: [type]
        lr: [learning rate]
        weight_decay: [weight decay]
        schedule: [learning rate schedule]
        # and other optimizer-specific configuration
      num_partition: [number of partitions]
      num_negative: [number of negative samples]
      batch_size: [batch size]
      episode_size: [episode size]

The number of partitions determines how to deal with multi-GPU or large graph cases.
The more partitions, the less GPU memory consumption and speed. The episode size
controls the synchronization frequency across partitions.

See section 3.2 in `GraphVite paper <GraphVite_>`_  for a detailed illustration.

.. code-block:: yaml

    train:
      model: [model]
      num_epoch: [number of epochs]
      negative_weight: [weight for negative sample]
      log_frequency: 1000
      # and other application-specific configuration

.. code-block:: yaml

    evaluate:
      - task: [task]
        # and other task-specific configuration
      - task: [task]
        ...

Evaluation is optional. There may be multiple evaluation tasks.

.. code-block:: yaml

    save:
        file_name: [file name]

Saving embeddings is optional.

For more detailed settings, we recommend to read the baseline configurations
for concrete examples. They can be found under ``config/`` in the Python package,
or in the `GitHub repository <Repo_>`_.

Global configuration
--------------------

We can overwrite the global settings of GraphVite in ``~/.graphvite/config.yaml``.

.. code-block:: yaml

    backend: [graphvite or torch]
    dataset_path: [path to store downloaded datasets]
    float_type: [default float type]
    index_type: [default index type]

By default, the evaluation backend is ``graphvite``. The datasets are stored in
``~/.graphvite/dataset``. The data types are ``float32`` and ``uint32`` respectively.