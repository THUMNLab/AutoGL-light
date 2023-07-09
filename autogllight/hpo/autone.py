"""
HPO Module for tuning hyper parameters
"""

import random
import time

import numpy as np
# from autogl.backend import DependentBackend
from tqdm import trange

# from ..feature import NetLSD as SgNetLSD
from . import register_hpo
from .autone_file import utils
from .base import BaseHPOptimizer, TimeTooLimitedError

# _isdgl = DependentBackend.is_dgl()
_isdgl = False
if _isdgl:
    import dgl
else:
    from torch_geometric.data import GraphSAINTRandomWalkSampler, InMemoryDataset

    class _MyDataset(InMemoryDataset):
        def __init__(self, datalist) -> None:
            super().__init__()
            self.data, self.slices = self.collate(datalist)


@register_hpo("autone")
class AutoNE(BaseHPOptimizer):
    """
    AutoNE HPOptimizer
    The Implementation of "AutoNE: Hyperparameter Optimization for Massive Network Embedding"(KDD 2019).
    See https://github.com/tadpole/AutoNE for more information

    Attributes
    ----------
    max_evals : int
        The max rounds of evaluating HPs
    subgraphs : int
        The number of subgraphs
    sub_evals : int
        The number of evaluation times on each subgraph
    sample_batch_size, sample_walk_length : int
        Using for sampling subgraph, see torch_geometric.data.GraphSAINRandomWalkSampler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get("max_evals", 100)
        self.subgraphs = kwargs.get("subgraphs", 2)
        self.sub_evals = kwargs.get("sub_evals", 2)
        self.sample_batch_size = kwargs.get("sample_batch_size", 150)
        self.sample_walk_length = kwargs.get("sample_walk_length", 100)
        self.dataset = kwargs.get("dataset", None)

    def optimize(self):
        """
        Optimize the HP by the method within give model and HP space

        See .base.BaseHPOptimizer.optimize
        """
        if self.dataset is None:
            raise ValueError("dataset is None")

        dataset = self.dataset

        space = self.hp_space
        current_space = self._encode_para(space)

        def sample_subgraph(whole_data):
            data = whole_data.data
            loader = GraphSAINTRandomWalkSampler(
                data,
                batch_size=self.sample_batch_size,
                walk_length=self.sample_walk_length,
                num_steps=self.subgraphs,
                save_dir=whole_data.processed_dir,
            )
            results = []
            for data in loader:
                in_dataset = _MyDataset([data])
                results.append(in_dataset)
            return results

        def sample_subgraph_dgl(whole_data):
            data = whole_data[0]  # dgl data
            # find data with different labels
            # random walk
            start = [
                random.randint(0, data.num_nodes - 1) for i in range(self.subgraphs)
            ]
            traces, _ = dgl.sampling.random_walk_with_restart(
                data,
                start,
                length=self.sample_batch_size,
                restart_prob=1 / self.sample_walk_length,
            )
            subgraphs = dgl.node_subgraph(data, [traces[i, :] for i in traces.size(0)])
            return subgraphs

        func = SgNetLSD()

        def get_wne(graph):
            graph = func.fit_transform(graph)
            # transform = nx.NxGraph.compose(map(lambda x: x(), nx.NX_EXTRACTORS))
            # print(type(graph))
            # gf = transform.fit_transform(graph).data.gf
            gf = graph.data.gf
            fin = list(gf[0]) + list(map(lambda x: float(x), gf[1:]))
            return fin

        start_time = time.time()

        # code in AutoNE
        sampled_number = self.subgraphs
        k = self.sub_evals
        s = self.max_evals
        X = []
        y = []
        params = utils.Params()
        params.set_space(current_space)
        total_t = 0.0
        info = []
        K = utils.K(len(params.type_))
        gp = utils.GaussianProcessRegressor(K)
        if _isdgl:
            sample_graphs = sample_subgraph_dgl(dataset)
        else:
            sample_graphs = sample_subgraph(dataset)
        print("Sample Phase:\n")
        for t in trange(sampled_number):
            b_t = time.time()
            i = t
            subgraph = sample_graphs[t]
            wne = get_wne(subgraph)
            for v in range(k):
                kargs = params.random_x()
                para = params.x2dict(kargs)
                externel_para, trial_para = self._decode_para(para)
                perf = self.f(externel_para, subgraph)
                X_reg = params.dict2x(trial_para)
                X.append(np.hstack((X_reg, wne)))
                y.append(perf)

        best_perf = None
        best_hp = None
        wne = get_wne(dataset)
        print("HPO Search Phase:\n")
        for t in trange(s):
            if time.time() - start_time > self.time_limit:
                self.logger.info("Time out of limit, Epoch: {}".format(str(i)))
                break
            b_t = time.time()
            gp.fit(np.vstack(X), y)
            X_temp, _ = gp.predict(params.get_bound(), params.get_type(), wne)
            X_temp = X_temp[: len(params.type_)]
            para = params.x2dict(X_temp)
            externel_para, trial_para = self._decode_para(para)
            perf_temp = self.f(externel_para, dataset)
            self.trials.append(
                self._creat_a_trail(
                    "HPO", "Autone", "Completed", externel_para, perf_temp
                )
            )
            # self._print_info(externel_para, perf_temp)
            X_reg = params.dict2x(trial_para)

            X.append(np.hstack((X_reg, wne)))
            y.append(perf_temp)
            if not best_perf or perf_temp < best_perf:
                best_perf = perf_temp
                best_hp = externel_para

            e_t = time.time()
            total_t += e_t - b_t

        if not best_perf:
            raise TimeTooLimitedError(
                "Given time is too limited to finish one round in HPO."
            )

        # self.logger.info("Best Parameter:")
        # self._print_info(best_hp, best_perf)

        return best_hp, best_perf
