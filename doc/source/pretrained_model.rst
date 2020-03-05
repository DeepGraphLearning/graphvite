Pre-trained Models
==================

.. include:: link.rst

To facilitate the usage of knowledge graph representations in semantic tasks, we
provide a bunch of pre-trained embeddings for some common datasets.

Wikidata5m
----------

`Wikidata5m`_ is a large-scale knowledge graph dataset constructed from `Wikidata`_
and `Wikipedia`_. It contains plenty of entities in the general domain, such as
celebrities, events, concepts and things.

We trained 5 standard knowledge graph embedding models on `Wikidata5m`_. The
performance benchmark of these models can be found :ref:`here <knowledge_graph_benchmark>`.

+-------------+-----------+---------+----------------------------+
| Model       | Dimension | Size    | Download link              |
+=============+===========+=========+============================+
| `TransE`_   | 512       | 9.33 GB | `transe_wikidata5m.pkl`_   |
+-------------+-----------+---------+----------------------------+
| `DistMult`_ | 512       | 9.33 GB | `distmult_wikidata5m.pkl`_ |
+-------------+-----------+---------+----------------------------+
| `ComplEx`_  | 512       | 9.33 GB | `complex_wikidata5m.pkl`_  |
+-------------+-----------+---------+----------------------------+
| `SimplE`_   | 512       | 9.33 GB | `simple_wikidata5m.pkl`_   |
+-------------+-----------+---------+----------------------------+
| `RotatE`_   | 512       | 9.33 GB | `rotate_wikidata5m.pkl`_   |
+-------------+-----------+---------+----------------------------+
| `QuatE`_    | 512       | 9.36 GB | `quate_wikidata5m.pkl`_    |
+-------------+-----------+---------+----------------------------+

.. _transe_wikidata5m.pkl: https://udemontreal-my.sharepoint.com/:u:/g/personal/zhaocheng_zhu_umontreal_ca/EX4c1Ud8M61KlDUn2U_yz_sBP_bXNuFnudfhRnYzWUFA2A?download=1
.. _distmult_wikidata5m.pkl: https://udemontreal-my.sharepoint.com/:u:/g/personal/zhaocheng_zhu_umontreal_ca/EQsXL8UmSJhHt2uBdB32muMBo4o4RUaMR6KDEQTcsz3jvg?download=1
.. _complex_wikidata5m.pkl: https://udemontreal-my.sharepoint.com/:u:/g/personal/zhaocheng_zhu_umontreal_ca/ERAwwLdsvdRIlrkVujMetmEBV9RgizsFnW91pIpjkBjbTw?download=1
.. _simple_wikidata5m.pkl: https://udemontreal-my.sharepoint.com/:u:/g/personal/zhaocheng_zhu_umontreal_ca/EVcJpJAzkThPu1vjgJLohscBgwtPajhTZvCCd8nEg1GiwA?download=1
.. _rotate_wikidata5m.pkl: https://udemontreal-my.sharepoint.com/:u:/g/personal/zhaocheng_zhu_umontreal_ca/EWvX5Z0rWZ9GvmdLaM3ONx4BtxzDFehXdc0gwE52YEiX2Q?download=1
.. _quate_wikidata5m.pkl: https://udemontreal-my.sharepoint.com/:u:/g/personal/zhaocheng_zhu_umontreal_ca/EUGNHMB9tlJAokjxBouyG08ByfAb3-IYHCszTMmJnQSegg?download=1

Load pre-trained models
-----------------------

The pre-trained models can be loaded through ``pickle``.

.. code-block:: python

    import pickle
    with open("transe_wikidata5m.pkl", "rb") as fin:
        model = pickle.load(fin)
    entity2id = model.graph.entity2id
    relation2id = model.graph.relation2id
    entity_embeddings = model.solver.entity_embeddings
    relation_embeddings = model.solver.relation_embeddings

Load the alias mapping from the dataset. Now we can access the embeddings by natural language index.

.. code-block:: python

    import graphvite as gv
    alias2entity = gv.dataset.wikidata5m.alias2entity
    alias2relation = gv.dataset.wikidata5m.alias2relation
    print(entity_embeddings[entity2id[alias2entity["machine learning"]]])
    print(relation_embeddings[relation2id[alias2relation["field of work"]]])