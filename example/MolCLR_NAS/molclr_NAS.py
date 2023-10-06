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
import shutil
import sys
import torch
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from nt_xent import NTXentLoss
from models.ginet_molclr import GINet
from models.gcn_molclr import GCN

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class MolCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self, trans_model, trans_optim):
        train_loader, valid_loader = self.dataset.get_data_loaders()
        model = trans_model
        model = self._load_pre_trained_weights(model)
        #print(model)
        
        optimizer = trans_optim
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (xis, xjs) in enumerate(train_loader):
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(epoch_counter, bn, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

            return best_valid_loss

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        return valid_loss

class NAS_top:
    def __init__(self):
        self.config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        print(self.config)
        if self.config['aug'] == 'node':
          from dataset.dataset import MoleculeDatasetWrapper
        elif self.config['aug'] == 'subgraph':
            from dataset.dataset_subgraph import MoleculeDatasetWrapper
        elif self.config['aug'] == 'mix':
            from dataset.dataset_mix import MoleculeDatasetWrapper
        else:
            raise ValueError('Not defined molecule augmentation!')

        self.dataset = MoleculeDatasetWrapper(self.config['batch_size'], **self.config['dataset'])

        self.molclr = MolCLR(self.dataset, self.config)

    def trainer(self, model, dataset, infer_mask, evaluation, *args, **kwargs):
        from autogllight.utils.backend.op import bk_mask, bk_label
        import torch.nn.functional as F
        
        # train and # validate
        mask = "train"
        device = next(model.parameters()).device
        model = GINet(**self.config["model"])
        dmodel = model.to(self.molclr.device)
        optim = torch.optim.Adam(dmodel.parameters(),lr = 1e-2, weight_decay=5e-7)
        #mask = bk_mask(dset, mask)
        ans_loss = self.molclr.train(dmodel, optim)
        
        return ans_loss
    
    def get_dataset(self):
        return self.dataset


os.environ['CUDA_VISIBLE_DEVICES'] = "2"

if __name__ == "__main__":
    set_seed(0)

    top = NAS_top()
    dataset = top.get_dataset().get_dataset_inner()
    input_dim = 512

    num_classes = 512

    space = SinglePathNodeClassificationSpace(
        input_dim=input_dim, output_dim=num_classes
    )
    space.instantiate()
    algo = RandomSearch(num_epochs = 10)
    
    estimator = TrainScratchEstimator(top.trainer)
    
    ans = algo.search(space, dataset, estimator)
    torch.save(ans, f"bb_model.syv")