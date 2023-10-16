.. _hpo_cn:

超参数优化
============================

我们支持不同的搜索空间下的黑盒超参数优化。

搜索空间
------------
我们支持如下三种搜索空间：

.. code-block:: python

    # 线性数值搜索空间：
    RangeHP("lr", 0.001, 0.1)

    # 对数数值搜索空间
    LogRangeHP("lr", 0.001, 0.1)

    # 类别搜索空间：
    ChoiceHP("opti", ["adam", "sgd"])

此外，用户还可以通过继承``HPSpace``来创建自己的搜索空间。搜索空间需要通过列表进行组合：

.. code-block:: python
    space = [
        LogRangeHP("lr", 0.001, 0.1),
        ChoiceHP("opti", ["adam", "sgd"]),
        ChoiceHP("num_layers", [2, 3]),
    ]


使用超参数优化器
--------------------

下表列出了我们支持的超参数优化算法及其所支持的搜索空间形式：

+-------------------+----------+----------+------+
|       算法         | 线性数值  | 对数数值  | 类别  |
+===================+==========+==========+======+
| Grid              |          |          | ✓    |
+-------------------+----------+----------+------+
| Random            | ✓        | ✓        | ✓    |
+-------------------+----------+----------+------+
| Anneal            | ✓        | ✓        | ✓    |
+-------------------+----------+----------+------+
| Bayes             | ✓        | ✓        | ✓    |
+-------------------+----------+----------+------+
| TPE [1]_          | ✓        | ✓        | ✓    |
+-------------------+----------+----------+------+
| CMAES [2]_        | ✓        | ✓        | ✓    |
+-------------------+----------+----------+------+
| MOCMAES [3]_      | ✓        | ✓        | ✓    |
+-------------------+----------+----------+------+
| Quasi random [4]_ | ✓        | ✓        | ✓    |
+-------------------+----------+----------+------+

我们使用如下方式使用超参数优化器：

.. code-block:: python
    def f(hps: dict, dataset=None):
        # return the result of evaluating the given hps on the given dataset
        ...
        return value

    hpo = build_hpo_from_name("tpe", hp_space, f)


添加你自己的超参数优化器（HPOptimizer）
--------------------

如果你想添加你自己的 HPOptimizer, 你需要实现``BaseHPOptimizer``这个接口:

.. code-block:: python

    class MyOptimizer(BaseHPOptimizer):
        def _set_up(self, num_slaves, time_limit, memory_limit):
            """
            Initialize something used in "optimize"

            Parameters
            ----------
            trainer : ..train.BaseTrainer
                Including model, giving HP space and using for training
            dataset : ...datasets
                Dataset to train and evaluate.
            time_limit : int
                Max time to run HPO
            memory_limit : None
                No implementation yet
            """
            ...

        def _update_trials(self, pid, hp, perf):
            """
            After the evaluation phase of each turn, update history trials according to the performance

            Parameters
            ----------
            pid : int
                The process id which runs the evaluation
            hp : dict
                The HPs used in evaluation
            perf : float
                The performance of the HP, higher is better
            """
            ...

        def _get_suggestion(self, pid):
            """
            Give the next HP suggestion

            Parameters
            ----------
            pid : int
                The process id which will run the evaluation

            Returns
            -------
            para_json: dict
                The suggested HP
            """
            ...

        def _best_hp(self):
            """
            Give the best HP and the best trainer as the returns of "optimize"

            Returns
            -------
            trainer: ..train.BaseTrainer
                The trainer including the best trained model
            para_json: dict
                The best HP
            """
            ...

        def optimize(self):
            """Key function. Return the best hp & performance"""
            # this functino can be omitted if using the default optimize()
            ...



.. [1] Bergstra, James S., et al. "Algorithms for hyper-parameter optimization." Advances in neural information processing systems. 2011.
.. [2] Arnold, Dirk V., and Nikolaus Hansen. "Active covariance matrix adaptation for the (1+ 1)-CMA-ES." Proceedings of the 12th annual conference on Genetic and evolutionary computation. 2010.
.. [3] Voß, Thomas, Nikolaus Hansen, and Christian Igel. "Improved step size adaptation for the MO-CMA-ES." Proceedings of the 12th annual conference on Genetic and evolutionary computation. 2010.
.. [4] Bratley, Paul, Bennett L. Fox, and Harald Niederreiter. "Programs to generate Niederreiter's low-discrepancy sequences." ACM Transactions on Mathematical Software (TOMS) 20.4 (1994): 494-495.