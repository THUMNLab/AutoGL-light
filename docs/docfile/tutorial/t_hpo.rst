.. _hpo:

Hyper Parameter Optimization
============================

We support black box hyper parameter optimization in various search spaces.

Search Space
------------

We support three types of search space as follows:

.. code-block:: python

    # linear numerical search space:
    RangeHP("lr", 0.001, 0.1)

    # logarithmic numerical search space:
    LogRangeHP("lr", 0.001, 0.1)

    # categorical search space:
    ChoiceHP("opti", ["adam", "sgd"])

In addition, users can also create their own search space by inheriting ``HPSpace``. The search space should be combined as a list:

.. code-block:: python
    space = [
        LogRangeHP("lr", 0.001, 0.1),
        ChoiceHP("opti", ["adam", "sgd"]),
        ChoiceHP("num_layers", [2, 3]),
    ]

Using HPOptimizer
--------------------

The following table lists the HPO algorithms we support and the search spaces they support:

+-------------------+------------------+-----------------------+-------------+
|     Algorithm     | linear numerical | logarithmic numerical | categorical |
+===================+==================+=======================+=============+
| Grid              |                  |                       | ✓           |
+-------------------+------------------+-----------------------+-------------+
| Random            | ✓                | ✓                     | ✓           |
+-------------------+------------------+-----------------------+-------------+
| Anneal            | ✓                | ✓                     | ✓           |
+-------------------+------------------+-----------------------+-------------+
| Bayes             | ✓                | ✓                     | ✓           |
+-------------------+------------------+-----------------------+-------------+
| TPE [1]_          | ✓                | ✓                     | ✓           |
+-------------------+------------------+-----------------------+-------------+
| CMAES [2]_        | ✓                | ✓                     | ✓           |
+-------------------+------------------+-----------------------+-------------+
| MOCMAES [3]_      | ✓                | ✓                     | ✓           |
+-------------------+------------------+-----------------------+-------------+
| Quasi random [4]_ | ✓                | ✓                     | ✓           |
+-------------------+------------------+-----------------------+-------------+

We use HPOptimizer as follows:

.. code-block:: python
    def f(hps: dict, dataset=None):
        # return the result of evaluating the given hps on the given dataset
        ...
        return value

    hpo = build_hpo_from_name("tpe", hp_space, f)

Adding Your HPOptimizer
--------------------
If you want to add your own HPOptimizer, you need to implement the ``BaseHPOptimizer`` interface:


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