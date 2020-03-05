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

Why is my CUDA driver version insufficient for CUDA runtime version?
--------------------------------------------------------------------

This is because you have installed a GraphVite compiled for some later CUDA version.
You can check your CUDA version with ``nvcc -V``, and then install the corresponding
package by

.. code-block:: bash

    conda install -c milagraph -c conda-forge graphvite cudatoolkit=x.x

where ``x.x`` is your CUDA version, e.g. 9.2 or 10.0.

Note graphvite does not support CUDA version earlier than 9.2, due to a failure of
old version ``nvcc``.

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

How can I speed up compliation?
-------------------------------

The compilation can be accelerated by reducing the number of template instantiations.
You can pass ``-DFAST_COMPILE=True`` to cmake, which will only compile commonly used
embedding dimensions. You may also comment out unnecessary instantiations in
``src/graphvite.cu`` for further speed-up.

How can I solve the BLAS issue in ``faiss``?
--------------------------------------------

``faiss`` is only required by the visualization application in GraphVite. If you do
not need visualization, you can pass ``-DNO_FAISS=True`` to cmake to skip that.