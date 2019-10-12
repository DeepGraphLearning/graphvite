Install
=======

GraphVite can be installed from either conda or source. You can also easily install
the library on `Google Colab`_ for demonstration.

.. _Google Colab: https://colab.research.google.com/

Install from conda
------------------

To install GraphVite from ``conda``, you only need one line.

.. code-block:: bash

    conda install -c milagraph graphvite cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")

By default, this will install all dependencies, including ``PyTorch`` and
``matplotlib``. If you only need embedding training without evaluation, there is an
alternative with minimum dependencies.

.. code-block:: bash

    conda install -c milagraph graphvite-mini cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")

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

Install on Colab
----------------

First, install Miniconda on Colab.

.. code-block:: bash

    !wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    !chmod +x Miniconda3-latest-Linux-x86_64.sh
    !./Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local -f

Then we install GraphVite and some tools for Jupyter Notebook.

.. code-block:: bash

    !conda install -y -c milagraph -c conda-forge graphvite \
        python=3.6 cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+\.\d+")
    !conda install -y wurlitzer ipykernel

Load the installed packages. Now you are ready to go.

.. code-block:: python

    import site
    site.addsitedir("/usr/local/lib/python3.6/site-packages")
    %reload_ext wurlitzer
