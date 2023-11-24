.. _nas:

神经架构搜索
============================

我们支持在不同搜索空间中应用不同的神经架构搜索算法。
神经架构搜索通常由三个模块构成: 搜索空间，搜索策略和评估策略。

搜索空间描述了所有可能被搜索的架构。空间主要由两部分组成，即操作（例如 GCNconv、GATconv）和输入-输出关系。
大的搜索空间可能有更好的最优架构，但需要更多的努力去探索。
人类的知识可以帮助设计一个合理的搜索空间，减少搜索策略的努力。

搜索策略控制了如何探索搜索空间，
它涵盖了经典的探索-利用权衡：
一方面，希望尽快找到性能良好的架构，
另一方面，应该避免过早地收敛到次优架构的区域。

评估策略在探索某一架构时给出其性能。
最简单的选项是在数据上对架构进行标准训练和验证。
由于在整个搜索过程中需要评估许多架构，评估策略需要非常高效以节约计算资源。

为了更灵活，我们将NAS过程模块化为三部分：搜索算法，搜索空间和评估器。
不同部分的不同模型可以在某些特定的约束下进行组合。
如果你想设计你自己的NAS过程，你可以根据你的需求改变其中任何一部分。


用法
-----

你可以直接引入特定的空间、算法、估计器来为特定的数据集搜索图神经网络。下面是一个例子：

.. code-block:: python
    
    from autogllight.nas.space import GraphNasNodeClassificationSpace
    from autogllight.nas.algorithm import GraphNasRL
    from autogllight.nas.estimator import OneShotEstimator
    from torch_geometric.datasets import Planetoid
    from os import path as osp
    import torch_geometric.transforms as T
    
    dataname = "cora"
    dataset = Planetoid(
        osp.expanduser("~/.cache-autogl"), dataname, transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    label = data.y
    input_dim = data.x.shape[-1]
    num_classes = len(np.unique(label.numpy()))

    space = GraphNasNodeClassificationSpace(input_dim=input_dim, output_dim=num_classes)
    space.instantiate()
    algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=False)
    estimator = OneShotEstimator()
    algo.search(space, dataset, estimator)

以上代码在搜索空间 ``GraphNasNodeClassificationSpace`` 使用 ``GraphNasRL`` 搜索算法进行搜索。

搜索空间
------------

搜索空间需要继承BaseSpace。
定义搜索空间主要有两种方式，一种可以以one-shot方式执行，另一种则不能。
目前，我们支持以下搜索空间： SinglePathNodeClassificationSpace, GassoSpace, GraphNasNodeClassificationSpace, GraphNasMacroNodeClassificationSpace, AutoAttendNodeClassificationSpace。

你也可以定义你自己的nas搜索空间。你应该重写``build_graph``函数来构建超网络。这里有一个例子。

.. code-block:: python

    from autogllight.space.base import BaseSpace

    # For example, create an NAS search space by yourself
    class SinglePathNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.2,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = ["gcn", "gat_8"],
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout

    def build_graph(self):
        for layer in range(self.layer_number):
            key = f"op_{layer}"
            in_dim = self.input_dim if layer == 0 else self.hidden_dim
            out_dim = (self.output_dim if layer == self.layer_number - 1 else self.hidden_dim)
            op_candidates = [
                op(in_dim, out_dim)
                if isinstance(op, type)
                else gnn_map(op, in_dim, out_dim)
                for op in self.ops
            ]
            self.setLayerChoice(layer, op_candidates, key=key)

    def forward(self, data):
        x = BK.feat(data)
        for layer in range(self.layer_number):
            op = getattr(self, f"op_{layer}")
            x = BK.gconv(op, data, x)
            if layer != self.layer_number - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

性能评估
---------------------

性能估计器用于估计架构的性能。目前我们支持以下估算器：

+-------------------------+-------------------------------------------------------+
| Estimator               | Description                                           |
+=========================+=======================================================+
| ``oneshot``             | 直接评估给定模型 |
+-------------------------+-------------------------------------------------------+
| ``scratch``             | 从头训练给定模型，再进行评估  |
+-------------------------+-------------------------------------------------------+

您也可以编写自己的估算器。这是一个在没有训练的情况下估计架构（用于one-shot空间）的示例。

.. code-block:: python

    # For example, create an NAS estimator by yourself
    from autogllight.nas.estimator.base import BaseEstimator
    class YourOneShotEstimator(BaseEstimator):
        # The only thing you should do is defining ``infer`` function
        def infer(self, model: BaseSpace, dataset, mask="train"):
            device = next(model.parameters()).device
            dset = dataset[0].to(device)
            # Forward the architecture
            pred = model(dset)[getattr(dset, f"{mask}_mask")]
            y = dset.y[getattr(dset, f'{mask}_mask')]
            # Use default loss function and metrics to evaluate the architecture
            loss = getattr(F, self.loss_f)(pred, y)
            probs = F.softmax(pred, dim = 1)
            metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
            return metrics, loss

搜索空间
---------------

空间策略定义了如何寻找架构。我们目前支持以下搜索策略：RandomSearch、Darts、RL、GraphNasRL、Enas、Spos、GRNA、Gasso。

不共享权重的基于样本的策略比具有权重共享的策略更简单。我们以DFS作为示例来展示如何定义自己的策略。


.. code-block:: python

    from autogllight.nas.algorithm.base import BaseNAS
    class RandomSearch(BaseNAS):
        # Get the number of samples at initialization
        def __init__(self, n_sample):
            super().__init__()
            self.n_sample = n_sample

        # The key process in NAS algorithm, search for an architecture given space, dataset and estimator
        def search(self, space: BaseSpace, dset, estimator):
            self.estimator=estimator
            self.dataset=dset
            self.space=space
                
            self.nas_modules = []
            k2o = get_module_order(self.space)
            # collect all mutables in the space
            replace_layer_choice(self.space, PathSamplingLayerChoice, self.nas_modules)
            replace_input_choice(self.space, PathSamplingInputChoice, self.nas_modules)
            # sort all mutables with given orders
            self.nas_modules = sort_replaced_module(k2o, self.nas_modules) 
            # get a dict cantaining all chioces
            selection_range={}
            for k,v in self.nas_modules:
                selection_range[k]=len(v)
            self.selection_dict=selection_range
                
            arch_perfs=[]
            # define DFS process
            self.selection = {}
            last_k = list(self.selection_dict.keys())[-1]
            def dfs():
                for k,v in self.selection_dict.items():
                    if not k in self.selection:
                        for i in range(v):
                            self.selection[k] = i
                            if k == last_k:
                                # evaluate an architecture
                                self.arch=space.parse_model(self.selection,self.device)
                                metric,loss=self._infer(mask='val')
                                arch_perfs.append([metric, self.selection.copy()])
                            else:
                                dfs()
                        del self.selection[k]
                        break
            dfs()

            # get the architecture with the best performance
            selection=arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
            arch=space.parse_model(selection,self.device)
            return arch 

不同的搜索策略应与不同的搜索空间和估算器结合使用。大多数搜索空间、搜索策略和估算器是兼容的。