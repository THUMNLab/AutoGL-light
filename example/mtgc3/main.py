'this file is the trainer of origin gnn'
import sys
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import numpy as np
from model import MTGC3
from tqdm import tqdm
import random

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

def train(model, device, loader, optimizer, arch_optimizer, stru_optimizer, task_type, eta, cris):
    model.train()
    currl_training = True
    use_xloss = False
    cls_criterion, reg_criterion = cris

    def get_loss(batch, pred):
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        if use_xloss:
            loss = cls_criterion(pred.to(torch.float32)[is_labeled].reshape(-1, 1), batch.y.to(torch.float32)[is_labeled].reshape(-1, 1))
        elif "classification" in task_type: 
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        else:
            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        return loss

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if arch_optimizer and step:
            model.ag.set = "notrain"

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            if arch_optimizer:
                arch_optimizer.zero_grad()
                stru_optimizer.zero_grad()
            pred = model(batch)
            loss = get_loss(batch, pred)
            loss.backward()
            ds = model.get_ds()
            if arch_optimizer:
                arch_optimizer.step()
                stru_optimizer.step()

            if currl_training:
                optimizer.zero_grad()
                arch_optimizer.zero_grad()
                pred = model.currl_forward(batch, ds, eta)
                loss = get_loss(batch, pred)
                loss.backward()
                ds = model.get_ds()

            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        #print(batch.y)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def get_args(): 
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=4,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--arch_optim', type=str, default="sgd",
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--arch_learning_rate', type=float, default=0.1,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--stru_learning_rate', type=float, default=0.02,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--alpha_temp', type=float, default=1.0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--mask_temp', type=float, default=0.3,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=132,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--stru_size', type=int, default=16,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--n_chunks', type=int, default=12,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--sep_head', type=bool, default=False,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-moltoxcast",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--pooling_ratio', type=float, default=0.5,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    args = parser.parse_args()
    return args

def main(args = None):
    if not args:
        args = get_args()
    args.temp = None
    args.loc_mean = None
    args.loc_std = None
    args.model_type = "darts"
    args.hidden_size = args.emb_dim
    args.num_layer = args.num_layers

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()
    cris = (cls_criterion, reg_criterion)

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset, root='~/tara/ogb/dataset')

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = MTGC3(None, dataset.num_features, dataset.num_tasks, args.emb_dim, args.num_layers, args.n_chunks, args.drop_ratio, epsilon=None,
                    args=args, with_conv_linear=False, num_nodes=-1, mol = True, virtual=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.arch_optim == 'sgd':
        arch_optimizer = optim.SGD(model.ag.parameters(), lr=args.arch_learning_rate)
    else:
        arch_optimizer = optim.Adam(model.ag.parameters(), lr=args.arch_learning_rate)

    stru_optimizer = torch.optim.SGD(model.supernet.stru_paras(), lr=args.stru_learning_rate) #fix lr in arch_optimizer

    valid_curve = []
    test_curve = []
    train_curve = []

    t1 = time.time()
    t0 = t1
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        print(time.time() - t1)
        t1 = time.time()
        
        if epoch % 10 == 3:
            print(t1 - t0)
            t0 = t1

        if arch_optimizer:
            if epoch == 90:
                model.ag.set = "paint"
            elif epoch % 20 == 19:
                model.ag.set = "train"
            else:
                model.ag.set = "notrain"
        
        eta = epoch / args.epochs
        model.supernet.eta = eta
        train(model, device, train_loader, optimizer, arch_optimizer, stru_optimizer, dataset.task_type, eta, cris)

        print('Evaluating...')
        #train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Validation': valid_perf, 'Test': test_perf})

        #train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        #scheduler.step()

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        #best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        #best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    return test_curve[best_val_epoch]

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch]}, args.filename)

def run(hps = {}):
    args = get_args()
    res = []
    for name in hps:
        setattr(args, name, hps[name])
    for i in range(1, 11):
        args.seed = i
        test_acc = main(args)
        res.append(test_acc)
    men, st = np.mean(res), np.std(res)
    print(men, st)
    return men, st

def test():
    # tox21
    hps = {'learning_rate': 0.001, 'weight_decay': 0.001, 'arch_optim': 'adam', 'arch_learning_rate': 0.03, 'alpha_temp': 1.4, 'mask_temp': 0.6, 'drop_ratio': 0.5}
    b = {'dataset': "ogbg-moltox21", 'emb_dim': 120, 'n_chunks': 12, 'sep_head': True}

    # toxcast
    #hps = {'learning_rate': 0.0025, 'weight_decay': 0, 'arch_optim': 'adam', 'arch_learning_rate': 0.02, 'alpha_temp': 2.4, 'mask_temp': 0.6, 'drop_ratio': 0.7}
    #b = {'dataset': "ogbg-moltoxcast", 'emb_dim': 128, 'n_chunks': 16, 'sep_head': False}    

    # sider
    hps = {'learning_rate': 0.0005, 'weight_decay': 0, 'arch_optim': 'sgd', 'arch_learning_rate': 0.025, 'alpha_temp': 0.2, 'mask_temp': 1.3, 'drop_ratio': 0.35}
    b = {'dataset': "ogbg-molsider", 'emb_dim': 27, 'n_chunks': 27, 'sep_head': True}    

    hps.update(b)
    run((hps))

if __name__ == "__main__":
    test()