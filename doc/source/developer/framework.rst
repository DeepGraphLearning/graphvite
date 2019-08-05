Understand the Framework
========================

The framework of GraphVite is composed of two parts, a core library and a Python wrapper.
The Python wrapper can be found in ``python/graphvite/``. It provides an auto wrapper for
classes in the core library, as well as implementation for applications and datasets.

The core library is implemented with C++11 and CUDA, and binded to Python using
`pybind11`_. It covers implementation of all computation-related classes in GraphVite,
such as graphs, solvers and optimizers. All these ingredients are packaged as classes,
similar to the Python interface. The source code can be found in ``include/`` and
``src/``.

.. _pybind11: https://pybind11.readthedocs.io

In the C++ implementation, there is something different from Python. The graphs and
solvers are templaterized by the underlying data types, and the length of embedding
vectors. This design enables dynamic data type in Python interface, as well as maximal
compile-time optimization.

The C++ interface is highly abstracted to faciliate further development on GraphVite.
Generally, by inheriting from the core interface, you can implement your graph deep
learning routine without caring about scheduling details.

The source code is organized as follows.

    - ``include/base/*`` implements basic data structures
    - ``include/util/*`` implements basic utils
    - ``include/core/*`` implements optimizers, and core interface of graphs and solvers
    - ``include/gpu/*`` implements forward & backward propagation for all models
    - ``include/instance/*`` implements instances of graphs and solvers
    - ``include/bind.h`` implements Python bindings
    - ``src/graphvite.cu`` instantiates all Python classes
