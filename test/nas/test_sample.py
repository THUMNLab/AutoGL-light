import os

os.environ["AUTOGL_BACKEND"] = "pyg"
import yaml
import random
import numpy as np
from autogllight.utils import *
from autogllight.nas.space import (
    SinglePathNodeClassificationSpace,
    GassoSpace,
    GraphNasNodeClassificationSpace,
    GraphNasMacroNodeClassificationSpace,
    AutoAttendNodeClassificationSpace,
)
from autogllight.nas.algorithm import (
    RandomSearch,
    Darts,
    RL,
    GraphNasRL,
    Enas,
    Spos,
    GRNA,
    Gasso,
)
from autogllight.nas.estimator import OneShotEstimator
from torch_geometric.datasets import Planetoid
from os import path as osp
import torch_geometric.transforms as T

if __name__ == "__main__":
    set_seed(0)

    dataname = "cora"
    dataset = Planetoid(
        osp.expanduser("~/.cache-autogl"), dataname, transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    label = data.y
    input_dim = data.x.shape[-1]
    num_classes = len(np.unique(label.numpy()))

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = RandomSearch(num_epochs=2)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = Darts(num_epochs=100)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = RL(num_epochs=2, ctrl_steps_aggregate=2, weight_share = True)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = RL(num_epochs=2, ctrl_steps_aggregate=2, weight_share = False)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share = True)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=False)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = Enas(num_epochs=2, ctrl_steps_aggregate=2)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = Spos(n_warmup=10, cycles = 200)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = SinglePathNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = GRNA(n_warmup=10, cycles = 200)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = GassoSpace(input_dim=input_dim, output_dim=num_classes)
    # space.instantiate()
    # algo = Gasso(num_epochs=20)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = GraphNasNodeClassificationSpace(input_dim=input_dim, output_dim=num_classes)
    # space.instantiate()
    # algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share = True)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = GraphNasMacroNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = GraphNasRL(num_epochs=2, ctrl_steps_aggregate=2, weight_share=False)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)

    # space = AutoAttendNodeClassificationSpace(
    #     input_dim=input_dim, output_dim=num_classes
    # )
    # space.instantiate()
    # algo = Spos(n_warmup=10, cycles = 200)
    # estimator = OneShotEstimator()
    # algo.search(space, dataset, estimator)
