Welcome to AutoGL-light's documentation!
==================================

AutoGL-light
------

*Actively under development by @THUMNLab*

A lightweight AutoML framework & toolkit for machine learning on graphs.

This is an extended package of `PyTorch <https://github.com/THUMNLab/AutoGL/>`_ .

Introduction
-----------

Our autogl-light library aims to serve automated graph machine learning and currently includes two main functionalities: graph hyperparameter optimization (HPO) and graph neural network architecture search (NAS). We plan to make this library compatible with various graph machine learning libraries as backends, but currently, we primarily support PyTorch Geometric. Compared to AutoGL, autogl-light does not fix the pipeline, i.e., it allows to freely incorporate graph HPO and graph NAS at any step of the workflow. We also expect autogl-light to be more user-friendly, especially for new users. 

Graph HPO aims to automatically optimize the hyperparameters of models in graph machine learning. Currently, we support algorithms such as Grid, Random, Anneal, Bayes, CAMES, MOCAMES, Quasi random, TPE, and AutoNE for hyperparameter optimization. For more details, please refer to `HPO documentations <http://mn.cs.tsinghua.edu.cn/AutoGL-light/docfile/tutorial/t_hpo.html>`_ .

Graph NAS aims to automatically design and optimize neural network architectures for graph machine learning. It searches for the optimal architecture within a given search space. Currently, we support search algorithms including Random, RL, EA, ENAS, SPOS, GraphNAS, DARTS, GRNA, GASSO, and GRACES. For more details, please refer to `NAS documentations <http://mn.cs.tsinghua.edu.cn/AutoGL-light/docfile/tutorial/t_nas.html>`_ .

To promote and showcase the usage of autogl-light, particularly in handlying various downstream graph tasks, we have included examples of applying autogl-light to bioinformatics using graph HPO and graph NAS, including `ScGNN <https://www.nature.com/articles/s41467-021-22197-x>`_ , `MolCLR <https://www.nature.com/articles/s42256-022-00447-x>`_ ,and `AutoGNNUQ <https://arxiv.org/abs/2307.10438>`_ Please refer to `example files <https://gitlink.org.cn/THUMNLab/AutoGL-light/tree/main/example>`_.

Installation
------------

Requirements
~~~~~~~~~~~~

Please make sure you meet the following requirements before installing AutoGL.

1. Python >= 3.6.0

2. PyTorch (>=1.6.0)

    see `PyTorch <https://pytorch.org/>`_ for installation.    

Installation
~~~~~~~~~~~~

Install from pip & conda
^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command to install this package through `pip`.

.. code-block:: shell

   pip install autogl-light


Install from source
^^^^^^^^^^^^^^^^^^^

Run the following command to install this package from the source.

.. code-block:: shell

   git clone https://www.gitlink.org.cn/THUMNLab/AutoGL-light.git
   cd AutoGL-light
   python setup.py install

Install for development
^^^^^^^^^^^^^^^^^^^^^^^


If you are a developer of the AutoGL-light project, please use the following command to create a soft link, then you can modify the local package without install them again.

.. code-block:: shell
   
   pip install -e .


Modules
-------

In AutoGLlight, the tasks are solved by corresponding modules, which in general do the following things:

1. Find the best suitable model architectures through neural architecture search. This is done by modules named **nas**. AutoGL provides several search spaces, algorithms and estimators for finding the best architectures.

2. Automatically train and tune popular models specified by users. This is done by modules named  **hyperparameter optimization**. 

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   docfile/tutorial/t_hpo
   docfile/tutorial/t_nas

.. toctree::
   :maxdepth: 2
   :caption: 中文教程

   docfile/tutorial_cn/t_hpo
   docfile/tutorial_cn/t_nas

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   docfile/documentation/hpo
   docfile/documentation/nas

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`