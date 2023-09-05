[English Introduction](README_en.md)

# 智图库轻量版

用于图数据的轻量版自动机器学习框架和工具包，是[智图库](https://www.gitlink.org.cn/THUMNLab/AutoGL)的扩展版本。

*由清华大学媒体与网络实验室进行开发与维护*

若有任何意见或建议，欢迎通过<a href="https://www.gitlink.org.cn/THUMNLab/AutoGL-light/issues">issues</a> 或邮件<a href="mailto:autogl@tsinghua.edu.cn">autogl@tsinghua.edu.cn</a> 与我们联系。


## 最新消息!

- 智图库轻量版第一个版本0.1.0发布! 
  - 支持图超参数优化（HPO）、神经架构搜索（NAS）
  - 提供了基于图算法处理生物数据的样例
	
## 介绍
 
我们的轻量版智图库旨在服务于自动图机器学习，目前主要包含图超参数优化和图神经网络架构搜索两个功能。我们计划该库未来可以支持使用不同的图机器学习库作为后端，但目前我们主要支持以PyTorch Geometric库作为后端。相比于智图库，轻量版智图库不要求使用固定的流水线模式，可以在支持在您的框架流程的任意步骤中自由地添加图超参数搜索和图神经网络架构搜索来辅助图机器学习任务。同时，我们旨在让轻量版智图库变得更为新用户友好。
                                        
### 图超参数优化
                                        
图超参数优化旨在对图机器学习模型中的超参数进行自动优化。目前我们支持了Grid，Random，Anneal，Bayes，CAMES，MOCAMES，Quasi random，TPE，AutoNE等算法。详情参见XXX。                          
                                        
### 图神经架构搜索
                                        
图神经架构搜索旨在对图机器学习中的神经网络模型架构进行自动设计与调优，它会在给定的架构搜索空间中自动搜索最优的图神经网络架构。目前我们支持了Random，RL, EA, ENAS, SPOS, GraphNAS, DARTS, GRNA, GASSO, GRACES等算法。详情参见XXX。
                                        
## 应用
                                       
为了增加轻量版智图库的易用性，特别是处理不同的下游图任务，我们在样例中添加了一些在生物领域图任务上采用轻量版智图库进行图超参数搜索和图神经网络架构搜索的例子。详情参加XXX。

## 安装

### 依赖

在安装轻量版智图库之前，请首先安装以下依赖项。

1. Python >= 3.6.0

2. PyTorch (>=1.6.0)

    详细信息请参考<https://pytorch.org/>。  

### 安装

#### 通过pip进行安装

运行以下命令以通过`pip`安装轻量版智图库。

```
pip install autogl-light
```

#### 从源代码安装
运行以下命令以从源安装智图。
```
git clone https://www.gitlink.org.cn/THUMNLab/AutoGL-light.git
cd AutoGL-light
python setup.py install
```

#### 开发者安装
如果您想以开发者方式安装轻量版智图，请运行以下命令以创建软链接，然后即可修改本地程序后而无需重复安装。
```
pip install -e .
```


## 引用

如果您使用了智图代码，请按如下方式引用我们的[论文](https://openreview.net/forum?id=0yHwpLeInDn)：
```
@inproceedings{guan2021autogl,
  title={Auto{GL}: A Library for Automated Graph Learning},
  author={Chaoyu Guan and Ziwei Zhang and Haoyang Li and Heng Chang and Zeyang Zhang and Yijian Qin and Jiyan Jiang and Xin Wang and Wenwu Zhu},
  booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
  year={2021},
  url={https://openreview.net/forum?id=0yHwpLeInDn}
}
```

或许您也会发现我们的[综述](http://arxiv.org/abs/2103.00742)有帮助:
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
轻量版智图库的所有代码采用[Apache license](LICENSE)。