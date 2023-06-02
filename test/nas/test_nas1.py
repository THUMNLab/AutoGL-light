import yaml
import random
import numpy as np
from autogllight.utils import *


if __name__ == "__main__":
    set_seed(0)
    
    dataname = 'cora'
    dataset = Planetoid(osp.expanduser('~/.cache-autogl'), dataname, transform=T.NormalizeFeatures())
    data = dataset[0]
    label = data.y
    num_classes = len(np.unique(label.numpy()))
    
    