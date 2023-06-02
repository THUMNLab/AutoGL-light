# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import OrderedDict

import time
import numpy as np
import torch


def get_hardware_aware_metric(model, hardware_metric):
    """
    Get architectures' hardware-aware metrics

    Attributes
    ----------
    model : BaseSpace
        The architecture to be evaluated
    hardware_metric : str
        The name of hardware-aware metric. Can be 'parameter' or 'latency'
    """

    if hardware_metric == "parameter":
        return count_parameters(model)
    elif hardware_metric == "latency":
        return measure_latency(model, 20, warmup_iters=5)
    else:
        raise ValueError("Unsupported hardware-aware metric")


def count_parameters(module, only_trainable=False):
    s = sum(
        p.numel()
        for p in module.parameters(recurse=False)
        if not only_trainable or p.requires_grad
    )
    if isinstance(module, PathSamplingLayerChoice):
        s += sum(count_parameters(m) for m in module.sampled_choices())
    else:
        s += sum(count_parameters(m) for m in module.children())
    return s


def measure_latency(model, num_iters=200, *, warmup_iters=50):
    device = next(model.parameters()).device
    num_feat = model.input_dim
    model.eval()
    latencys = []
    data = _build_random_data(device, num_feat)
    with torch.no_grad():
        try:
            for i in range(warmup_iters + num_iters):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                model(data)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - start
                if i >= warmup_iters:
                    latencys.append(dt)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                INF = 100
                return INF
            else:
                raise e

    return np.mean(latencys)


def _build_random_data(device, num_feat):
    node_nums = 3000
    edge_nums = 10000

    class Data:
        pass

    data = Data()
    data.x = torch.randn((node_nums, num_feat)).to(device)
    data.edge_index = torch.randint(0, node_nums, (2, edge_nums)).to(device)
    data.num_features = num_feat
    return data


def to_device(obj, device):
    """
    Move a tensor, tuple, list, or dict onto device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def to_list(arr):
    if torch.is_tensor(arr):
        return arr.cpu().numpy().tolist()
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return arr


class AverageMeterGroup:
    """
    Average meter group for multiple average meters.
    """

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        """
        Update the meter group with a dict of metrics.
        Non-exist average meters will be automatically created.
        """
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        """
        Return a summary string of group data.
        """
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """
    Computes and stores the average and current value.

    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.

        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)
