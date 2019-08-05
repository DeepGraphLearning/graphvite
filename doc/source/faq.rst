Frequently Asked Questions
==========================

.. contents::
    :local:

How should I cite GraphVite?
----------------------------

If you find GraphVite helps your research, please cite it in your publications.

.. code-block:: none

    @inproceedings{zhu2019graphvite,
        title={GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding},
        author={Zhu, Zhaocheng and Xu, Shizhen and Qu, Meng and Tang, Jian},
        booktitle={The World Wide Web Conference},
        pages={2494--2504},
        year={2019},
        organization={ACM}
    }

Why is there a ``PackagesNotFoundError`` in conda installation?
---------------------------------------------------------------

Some dependencies of the library aren't present in the default channels of conda.
Config your conda with the following command, and try installation again.

.. code-block:: bash

    conda config --add channels conda-forge

Why is there a compilation error for template deduction?
--------------------------------------------------------

This is due to a failure of old version ``nvcc`` in compiling the templates in
``pybind11``. Generally, ``nvcc 9.2`` or later will work.

Why is the access to embeddings so slow?
----------------------------------------

Due to the binding mechanism, the numpy view of embeddings is generated each time
when you access the embeddings in Python. Such generation may take a non-trivial
overhead. To avoid that cost, we recommend you to copy the reference of the
embeddings.

.. code-block:: python

    embeddings = solver.vertex_embeddings

Now the access to ``embeddings`` should be good.