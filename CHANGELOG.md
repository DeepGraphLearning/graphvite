Change log
==========

Here list all notable changes in GraphVite library.

v0.2.2 - 2020-03-11
-------------------
- New model QuatE and its benchmarks on 5 knowledge graph datasets.
- Add an option to skip `faiss` in compilation.
- Fix instructions for conda installation.

v0.2.1 - 2019-11-12
-------------------
- New dataset `Wikidata5m` and its benchmarks,
  including TransE, DistMult, ComplEx, SimplE and RotatE.
- Add interface for loading pretrained models and save hyperparameters.
- Add weight clip in asynchronous self-adversarial negative sampling.

v0.2.0 - 2019-10-11
-------------------
- Add scalable multi-GPU prediction for node embedding and knowledge graph embedding.
  Evaluation on link prediction is 4.6x faster than v0.1.0.
- New demo dataset `math` and entity prediction evaluation for knowledge graph.
- Support Kepler and Turing GPU architectures.
- Automatically choose the best episode size with regrad to RAM limit.
- Add template config files for applications.
- Change the update of global embeddings from average to accumulation. Fix a serious
  numeric problem in the update.
- Move file format settings from graph to application. Now one can customize formats
  and use comments in evaluation files. Add document for data format.
- Separate GPU implementation into training routines and models. Routines are in
  `include/instance/gpu/*` and models are in `include/instance/model/*`.

v0.1.0 - 2019-08-05
-------------------
- Multi-GPU training of large-scale graph embedding 
- 3 applications: node embedding, knowledge graph embedding and graph &
  high-dimensional data visualization
- Node embedding
    - Model: DeepWalk, LINE, node2vec
    - Evaluation: node classification, link prediction
- Knowledge graph embedding
    - Model: TransE, DistMult, ComplEx, SimplE, RotatE
    - Evaluation: link prediction
- Graph & High-dimensional data visualization
    - Model: LargeVis
    - Evaluation: visualization(2D / 3D), animation(3D), hierarchy(2D)