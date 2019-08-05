Quick Start
===========

Here is a quick-start example that illustrate the pipeline in GraphVite. If you don't
have ``pytorch`` installed, simply add ``--no-eval`` to skip the evaluation stage.

.. code-block:: bash

    graphvite baseline quick start

The example will automatically download a social network dataset called BlogCatalog,
where nodes correspond to blog users. For each node, we learn an embedding vector,
and evaluate the embeddings by using them as features for multi-label node
classifcation.

Typically, the example takes no more than 1 minute. You will obtain some output like

.. code-block:: none

    Batch id: 6000
    loss = 0.371641

    macro-F1@20%: 0.236794
    micro-F1@20%: 0.388110

Note that the F1 scores may vary across different trials,
as only one random split is evaluated for quick demonstration here.

The learned embeddings are saved into a compressed numpy dump.
You can load them for further use

    >>> import pickle
    >>> with open("line_blogcatalog.pkl", "rb") as fin:
    >>>     blogcatalog = pickle.load(fin)
    >>> names = blogcatalog.id2name
    >>> embeddings = blogcatalog.vertex_embeddings
    >>> print(names[1024], embddings[1024])

As the embeddings might be further used in other downstream tasks, it would be
helpful if they can be obtained in the easiest way.

For a more in-depth tutorial about GraphVite, take a look at

- :doc:`user/command_line`
- :doc:`user/configuration`
- :doc:`user/python`
- :doc:`user/auto`