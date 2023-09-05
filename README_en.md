[中文介绍](../..)

# Lightweight Auto Graph Learning

A lightweight AutoML framework & toolkit for machine learning on graphs.

This is an extended package of [AutoGL](https://github.com/THUMNLab/AutoGL).

*Actively under development by @THUMNLab*

Feel free to open <a href="https://www.gitlink.org.cn/THUMNLab/AutoGL-light/issues">issues</a> or contact us at <a href="mailto:autogl@tsinghua.edu.cn">autogl@tsinghua.edu.cn</a> if you have any comments or suggestions!


## News!
- We have released the first version 0.1.0! 
    - We support Hyper-parameter Optimization (HPO) and Neural Architecture Search (NAS). 
    - We also provide showcases for using graph machine learning for bioinformatics.
	
## Introduction
 Our autogl-light library aims to serve automated graph machine learning and currently includes two main functionalities: graph hyperparameter optimization (HPO) and graph neural network architecture search (NAS). We plan to make this library compatible with various graph machine learning libraries as backends, but currently, we primarily support PyTorch Geometric. Compared to AutoGL, autogl-light does not fix the pipeline, i.e., it allows to freely incorporate graph HPO and graph NAS at any step of the workflow. We also expect autogl-light to be more user-friendly, especially for new users. 。
                                        
### Graph Hyper-parameter Optimization                                        
Graph HPO aims to automatically optimize the hyperparameters of models in graph machine learning. Currently, we support algorithms such as Grid, Random, Anneal, Bayes, CAMES, MOCAMES, Quasi random, TPE, and AutoNE for hyperparameter optimization. For more details, please refer to XXX.              
                                        
### Graph Neural Architecture Search                                        
Graph NAS aims to automatically design and optimize neural network architectures for graph machine learning. It searches for the optimal architecture within a given search space. Currently, we support search algorithms including Random, RL, EA, ENAS, SPOS, GraphNAS, DARTS, GRNA, GASSO, and GRACES. For more details, please refer to XXX.
                                        
## Applications
To promote and showcase the usage of autogl-light, particularly in handlying various downstream graph tasks, we have included examples of applying autogl-light to bioinformatics using graph HPO and graph NAS. For more information, please refer to XXX.

## Installation
### Requirements

Please make sure you meet the following requirements before installing AutoGL.

1. Python >= 3.6.0

2. PyTorch (>=1.6.0)

    see <https://pytorch.org/> for installation.    

### Installation

#### Install from pip

Run the following command to install this package through `pip`.

```
pip install autogl-light
```

#### Install from source

Run the following command to install this package from the source.

```
git clone https://www.gitlink.org.cn/THUMNLab/AutoGL-light.git
cd AutoGL-light
python setup.py install
```

#### Install for development

If you are a developer of the AutoGL-light project, please use the following command to create a soft link, then you can modify the local package without install them again.

```
pip install -e .
```


## Cite

Please cite [our paper](https://openreview.net/forum?id=0yHwpLeInDn) as follows if you find our code useful:
```
@inproceedings{guan2021autogl,
  title={Auto{GL}: A Library for Automated Graph Learning},
  author={Chaoyu Guan and Ziwei Zhang and Haoyang Li and Heng Chang and Zeyang Zhang and Yijian Qin and Jiyan Jiang and Xin Wang and Wenwu Zhu},
  booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
  year={2021},
  url={https://openreview.net/forum?id=0yHwpLeInDn}
}
```

You may also find our [survey paper](http://arxiv.org/abs/2103.00742) helpful:
```
@article{zhang2021automated,
  title={Automated Machine Learning on Graphs: A Survey},
  author={Zhang, Ziwei and Wang, Xin and Zhu, Wenwu},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
  year={2021},
  note={Survey track}
}
```

## License
We follow [Apache license](LICENSE) across the entire codebase.
