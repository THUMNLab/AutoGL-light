import os

os.environ["AUTOGL_BACKEND"] = "pyg"
import yaml
import random
import numpy as np
from autogllight.utils import *
from autogllight.nas.space import (
    SinglePathNodeClassificationSpace,
)
from autogllight.nas.algorithm import (
    RandomSearch,
)
from autogllight.nas.estimator import OneShotEstimator, TrainScratchEstimator
from torch_geometric.datasets import Planetoid
from os import path as osp
import torch_geometric.transforms as T
from tqdm import trange

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

    space = SinglePathNodeClassificationSpace(
        input_dim=input_dim, output_dim=num_classes
    )
    space.instantiate()
    algo = RandomSearch(num_epochs=2)
    
    def trainer(model, dataset, infer_mask, evaluation, *args, **kwargs):
        from autogllight.utils.backend.op import bk_mask, bk_label
        import torch.nn.functional as F
        
        # train
        mask = "train"
        optim = torch.optim.Adam(model.parameters(),lr = 1e-2, weight_decay=5e-7)
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        mask = bk_mask(dset, mask)
        epochs = 100
        with trange(epochs) as bar:
            for e in bar:
                pred = model(dset, *args, **kwargs)[mask]
                label = bk_label(dset)
                y = label[mask]
                loss = getattr(F, 'nll_loss')(pred, y)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
        
        # infer
        mask = infer_mask
        dset = dataset[0].to(device)
        mask = bk_mask(dset, mask)
        pred = model(dset, *args, **kwargs)[mask]
        label = bk_label(dset)
        y = label[mask]
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()
        y = y.cpu()
        metrics = {
            eva.get_eval_name(): eva.evaluate(probs, y) for eva in evaluation
        }
        return metrics, loss.item()
    
    estimator = TrainScratchEstimator(trainer)
    
    algo.search(space, dataset, estimator)