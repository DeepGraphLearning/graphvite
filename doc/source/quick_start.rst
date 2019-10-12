Quick Start
===========

Here is a quick-start example that illustrate the pipeline in GraphVite. If ``pytorch``
is not installed, we can simply add ``--no-eval`` to skip the evaluation stage.

.. code-block:: bash

    graphvite baseline quick start

The example will automatically download a social network dataset called BlogCatalog,
where nodes correspond to blog users. For each node, we learn an embedding vector that
preserves its neighborhood structure, which is done by minimizing a reconstruction
loss. GraphVite will display the progress and the loss during training.

Once the training is done, the learned embeddings are evaluated on link prediction and
node classification tasks. For link prediction, we try to predict unseen edges with
the embeddings. For node classification, we use the embeddings as inputs for
multi-label classification of nodes.

Typically, this example takes no more than 1 minute. We will obtain some output like

.. code-block:: none

    Batch id: 6000
    loss = 0.371041

    ------------- link prediction --------------
    AUC: 0.899933
    
    ----------- node classification ------------
    macro-F1@20%: 0.242114
    micro-F1@20%: 0.391342

Note that the F1 scores may vary across different trials, as only one random split is
evaluated for quick demonstration here.

The learned embeddings are saved into a pickle dump. We can load them for further
use.

    >>> import pickle
    >>> with open("line_blogcatalog.pkl", "rb") as fin:
    >>>     blogcatalog = pickle.load(fin)
    >>> names = blogcatalog.id2name
    >>> embeddings = blogcatalog.vertex_embeddings
    >>> print(names[1024], embeddings[1024])

Another interesting example is a synthetic math dataset of arithmetic operations. By
treating the operations as relations of a knowledge graph, we can learn embeddings
that generalize to unseen triplets (i.e. computation formulas). Check out this example
with

.. code-block:: bash

    graphvite baseline math

For a more in-depth tutorial about GraphVite, take a look at

- :doc:`user/command_line`
- :doc:`user/configuration`
- :doc:`user/python`
- :doc:`user/auto`