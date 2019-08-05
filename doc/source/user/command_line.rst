Command Line
============

As you see in :doc:`../quick_start`, GraphVite can be simply invoked from a command
line. Here are some other useful commands you may use.

Reproduce baseline benchmarks
-----------------------------

.. code-block:: bash

    graphvite baseline [keyword ...] [--no-eval] [--gpu n] [--cpu m]

GraphVite provides a large number of baselines on standard datasets. To reproduce
a baseline benchmark, you only need to specify the keywords of the experiment, and
the library will do all the rest.

By default, baselines are configured to use all CPUs and GPUs. You may override this
behavior by specifying the number of GPUs and the number of CPUs per GPU.

For example, the following command line reproduces RotatE model on FB15k dataset,
using 4 GPUs and 12 CPUs.

.. code-block:: bash

    graphvite baseline rotate fb15k --gpu 4 --cpu 3

Use ``graphvite list`` to get a list of available baselines.

Run configuration files
-----------------------

.. code-block:: bash

    graphvite run [config] [--no-eval] [--gpu n] [--cpu m]

Experiments can be easily conducted in GraphVite by specifying a yaml configuration.
For how to write an experiment configuration, see :doc:`configuration`.

Visualize high-dimensional vectors
----------------------------------

.. code-block:: bash

    graphvite visualize [file] [--label label_file] [--save save_file] [--perplexity n] [--3d]

You can visualize your high-dimensional vectors with a simple command line in
GraphVite.

The file can be either in numpy dump or text format. You can also provide a label
file indicating the category of each data point. For the save file, we recommend to
use a ``png`` format, while ``pdf`` is also supported.