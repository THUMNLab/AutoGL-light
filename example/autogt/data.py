import time
import torch
import pyximport
import numpy as np
import os.path as osp
import torch_geometric
from scipy import sparse
from collator import collator
from functools import partial
from scipy import sparse as sp
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos


def get_dataset(args, split=0):
    start = time.time()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'TU')
    dataset = torch_geometric.datasets.TUDataset(path, args.dataset_name)
    size = len(dataset)
    n = np.zeros(size, dtype=int)
    X = np.zeros(size)
    Y = np.zeros(size)
    for i, item in enumerate(dataset):
        n[i] = i
        Y[i] = item.y
    skf = StratifiedKFold(10)
    sss = StratifiedShuffleSplit(1, test_size=1/9, random_state=42)
    train_split = []
    valid_split = []
    test__split = []
    for train_index, test_index in skf.split(X, Y):
        test__split.append(test_index)
        n_train = n[train_index]
        X_train = X[train_index]
        Y_train = Y[train_index]
        for train_index, valid_index in sss.split(X_train, Y_train):
            train_split.append(n_train[train_index])
            valid_split.append(n_train[valid_index])
    dataset = MyTUDataset(path, name=args.dataset_name, args=args)
    train = np.array(train_split[split], dtype=int)
    valid = np.array(valid_split[split], dtype=int)
    test_ = np.array(test__split[split], dtype=int)
    train_dataloader = DataLoader(dataset[train], args.batch_size
        , num_workers=8, collate_fn=partial(collator, max_node=args.max_node))
    valid_dataloader = DataLoader(dataset[valid], args.batch_size
        , num_workers=8, collate_fn=partial(collator, max_node=args.max_node))
    test_dataloader_ = DataLoader(dataset[test_],  args.batch_size
        , num_workers=8, collate_fn=partial(collator, max_node=args.max_node))
    train_loader = [len(train_dataloader.dataset)]
    valid_loader = [len(valid_dataloader.dataset)]
    test_loader_ = [len(test_dataloader_.dataset)]
    for batched_data in train_dataloader:
        train_loader.append(batched_data.to(torch.device('cuda')))
    for batched_data in valid_dataloader:
        valid_loader.append(batched_data.to(torch.device('cuda')))
    for batched_data in test_dataloader_:
        test_loader_.append(batched_data.to(torch.device('cuda')))
    end = time.time()
    print(f'{args.dataset_name} Dataset Loaded! Use time: {end - start} s')
    return [train_loader, valid_loader, test_loader_]


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item, args):
    edge_attr = torch.LongTensor(item.edge_attr.numpy())
    edge_index = torch.LongTensor(item.edge_index.numpy())
    x = torch.LongTensor(item.x.numpy())
    N = x.size(0)
    x = convert_to_single_emb(x, 2)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr, 2) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    # Laplacian Eigenvectors
    A = adj.numpy().astype(float)
    in_degree = item.in_degree
    in_degree[in_degree == 0] = 1
    N = sp.diags(in_degree.numpy() ** -0.5, dtype=float)
    L = sp.eye(adj.shape[0]) - N * A * N

    lap_enc_dim = args.lap_enc_dim
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_pos_enc = EigVec[:, 1:lap_enc_dim+1]
    pad_dim = max(lap_enc_dim - lap_pos_enc.shape[1], 0)
    lap_pos_enc = np.pad(lap_pos_enc, ((0, 0), (0, pad_dim)), 'constant', constant_values=(0, 0))
    item.lap_pos_enc = torch.from_numpy(lap_pos_enc).float()

    # Singular Value Decomposition
    svd_enc_dim = args.svd_enc_dim
    svd_dim = int(svd_enc_dim / 2)
    k = int(min(svd_dim, A.shape[0] - 1))
    u, s, vh = sparse.linalg.svds(A, k)
    sqrts = np.diag(s ** 0.5)
    u = np.pad(u.dot(sqrts), ((0, 0), (0, svd_dim - k)), 'constant', constant_values=(0, 0))
    v = np.pad(vh.T.dot(sqrts), ((0, 0), (0, svd_dim - k)), 'constant', constant_values=(0, 0))
    svd = np.concatenate((u, v), axis=1)
    item.svd_pos_enc = torch.from_numpy(svd).to(torch.float32)

    # Proximity-Enhanced Multi-Head Attention
    A_tilde = A + np.diag(np.ones(A.shape[0]))
    N = sp.diags((item.in_degree.view(-1).numpy() + 1) ** -0.5, dtype=float)
    A_tilde = N * A_tilde * N
    A_tilde = torch.Tensor(A_tilde)
    tilde_list = [torch.matrix_power(A_tilde, i).unsqueeze(-1) for i in range(args.pma_dim)]
    pma = torch.cat(tilde_list, dim=-1)
    item.pma_att_enc = pma
    return item


class MyTUDataset(torch_geometric.datasets.TUDataset):
    def __init__(self, path, name, args):
        super(MyTUDataset, self).__init__(path, name)
        self.args = args

    def download(self):
        super(MyTUDataset, self).download()

    def process(self):
        super(MyTUDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            if hasattr(item, "edge_attr") == False or item.edge_attr == None:
                item.edge_attr = torch.zeros((item.edge_index.shape[1], 1))
            return preprocess_item(item, self.args)
        else:
            return self.index_select(idx)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")
        parser.add_argument('--lap_enc_dim', type=int, default=10)
        parser.add_argument('--svd_enc_dim', type=int, default=16)
        parser.add_argument('--pma_dim', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=32)
        return parent_parser


def check_dataset(dataset_name):
    start = time.time()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'TU')
    # dataset = MyTUDataset(path, name='PROTEINS')
    dataset = torch_geometric.datasets.TUDataset(path, name=dataset_name)
    print("Dataset size: {}.".format(len(dataset)))
    print("Dataset example: {}.".format(dataset[0]))
    max_node = 0
    max_x_offset = 0
    max_y_offset = 0
    max_edge_attr_offset = 0
    no_edge_count = 0
    for item in dataset:
        if max_node < item.x.shape[0]:
            max_node = item.x.shape[0]
        if max_x_offset < torch.max(item.x):
            max_x_offset = torch.max(item.x)
        if max_y_offset < torch.max(item.y):
            max_y_offset = torch.max(item.y)
        if item.edge_index.shape[1] == 0:
            no_edge_count += 1
        elif item.edge_attr != None and max_edge_attr_offset < torch.max(item.edge_attr):
            max_edge_attr_offset = torch.max(item.edge_attr)
            print(item)
    print("Max number of node is {}.".format(max_node))
    print("Max offset of x is {}.".format(max_x_offset))
    print("Max offset of y is {}.".format(max_x_offset))
    print("Max offset of edge_attr is {}.".format(max_edge_attr_offset))
    print("There are {} graph without node.".format(no_edge_count))
    end = time.time()
    print("Program Ended! Use time: {} s".format(end - start))


def split_test():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'TU')
    dataset = torch_geometric.datasets.TUDataset(path, 'PROTEINS')
    size = len(dataset)
    n = np.zeros(size, dtype=int)
    for i in range(0, size):
        n[i] = i
    X = np.zeros(size)
    Y = np.zeros(size)
    print(dataset)
    for i, item in enumerate(dataset):
        Y[i] = item.y
    skf = StratifiedKFold(10)
    sss = StratifiedShuffleSplit(1, test_size=1/9, random_state=0)
    train_split = []
    valid_split = []
    test__split = []
    for train_index, test_index in skf.split(X, Y):
        test__split.append(test_index)
        n_train = n[train_index]
        X_train = X[train_index]
        Y_train = Y[train_index]
        for train_index, valid_index in sss.split(X_train, Y_train):
            train_split.append(n_train[train_index])
            valid_split.append(n_train[valid_index])
    return train_split, valid_split, test__split
