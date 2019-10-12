Command Line
============

As we have seen in :doc:`../quick_start`, GraphVite can be simply invoked from a
command line. Here are some other useful commands we can use.

Reproduce baseline benchmarks
-----------------------------

.. code-block:: bash

    graphvite baseline [keyword ...] [--no-eval] [--gpu n] [--cpu m] [--epoch e]

GraphVite provides a large number of baselines on standard datasets. To reproduce
a baseline benchmark, we only need to specify the keywords of the experiment, and
the library will do the rest for us.

By default, baselines are configured to use all CPUs and GPUs. We may override this
behavior by specifying the number of GPUs and the number of CPUs per GPU. We may also
override the number of training epochs for fast experiments.

For example, the following command line reproduces RotatE model on FB15k dataset,
using 4 GPUs and 12 CPUs.

.. code-block:: bash

    graphvite baseline rotate fb15k --gpu 4 --cpu 3

Use ``graphvite list`` to get a list of available baselines.

Run configuration files
-----------------------

Custom experiments can be easily carried out in GraphVite through a yaml configuration.
This is especially convenient if we want to use GraphVite as an off-the-shelf tool
for pretraining embeddings.

.. code-block:: bash

    graphvite new [application ...] [--file f]

The above command creates a configuration scaffold for our application, where most
settings are ready. We just need to fill a minimal number of settings following the
instructions. For a more detailed introduction on configuration files, see
:ref:`experiment configuration`.

Once we complete the configuration file, we can run it by

.. code-block:: bash

    graphvite run [config] [--no-eval] [--gpu n] [--cpu m] [--epoch e]

Visualize high-dimensional vectors
----------------------------------

.. code-block:: bash

    graphvite visualize [file] [--label label_file] [--save save_file] [--perplexity n] [--3d]

We can visualize our high-dimensional vectors with a simple command line in
GraphVite.

The file can be either a numpy dump ``*.npy`` or a text matrix ``*.txt``. We can
also provide a label file indicating the category of each data point. For the save
file, we recommend to use ``png`` format, while ``pdf`` is also supported.