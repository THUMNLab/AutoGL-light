# Lightweight Auto Graph Learning

A lightweight AutoML framework & toolkit for machine learning on graphs.

This is an extended package of [AutoGL](https://github.com/THUMNLab/AutoGL).

*Actively under development by @THUMNLab*

Feel free to open <a href="https://github.com/THUMNLab/AutoGL-light/issues">issues</a> or contact us at <a href="mailto:autogl@tsinghua.edu.cn">autogl@tsinghua.edu.cn</a> if you have any comments or suggestions!


## News!

- We have released the first version 0.1.0!
	
## Introduction
 
我们的智图-light开源库服务于自动图学习，其中包含图超参数优化和图神经网络架构搜索两个功能。我们的库支持使用不同的图机器学习库作为后端。目前，我们主要支持以PyTorch Geometric库作为后端。相比智图，智图light没有要求使用某种固定的流水线模式。您可以在您的框架流程的任意步骤中自由地添加图超参数搜索和图神经网络架构搜索来辅助机器学习任务。
                                        
### 图超参数优化
                                        
图超参数优化旨在对图机器学习中的超参数进行自动优化，目前我们支持了 Grid，Random，Anneal，Bayes，CAMES，MOCAMES，Quasi random，TPE，AutoNE等算法。详情参见。                          
                                        
### 图神经网络架构搜索
                                        
图神经网络架构搜索旨在对图机器学习中的模型架构进行调优，它会在给定的架构搜索空间中自动搜索最好的图神经网络架构。目前我们支持了 Random， RL,  EA, ENAS, SPOS, GraphNAS, DARTS, GRNA, GASSO, GRACES等算法。详情参见。
                                        
## 应用
                                       
为了更好地体现智图-light的易用性，我们在example中添加了一些在生物领域图数据集的任务上采用智图-light进行图超参数搜索和图神经网络架构搜索的例子。

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
git clone https://github.com/THUMNLab/AutoGL-light.git
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
@inproceedings{
guan2021autogl,
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
