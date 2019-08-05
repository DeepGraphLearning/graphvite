Install
=======

There are 2 ways to install GraphVite.

Install from conda
------------------

You can install GraphVite from ``conda`` with only one line.

.. code-block:: bash

    conda install -c milagraph graphvite

By default, this will install all dependencies, including ``PyTorch`` and
``matplotlib``. If you only need embedding training without evaluation, you can take
the following alternative with minimum dependencies.

.. code-block:: bash

    conda install -c milagraph graphvite-mini

Install from source
-------------------

First, clone GraphVite from GitHub.

.. code-block:: bash

    git clone https://github.com/DeepGraphLearning/graphvite
    cd graphvite

Install compilation and runtime dependencies via ``conda``.

.. code-block:: bash

    conda install -y --file conda/requirements.txt

Compile the code using the following directives. If you have ``faiss`` installed
from source, you can pass ``-DFAISS_PATH=/path/to/faiss`` to ``cmake``.

.. code-block:: bash

    mkdir build
    cd build && cmake .. && make && cd -

Finally, install Python bindings.

.. code-block:: bash

    cd python && python setup.py install && cd -