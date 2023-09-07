import torch
import torch_geometric.nn as pygnn
import torch.nn.functional as F
from .supernet.conv import Conv

COAUTHOR = [
    'gatv2',
    'gcn',
    'sage',
    'linear',
    'gin',
    'graph'
]

class Supernet(torch.nn.Module):
    def __init__(self, n_layers, n_input, n_output, n_hidden, dropout, space=COAUTHOR, arch=None, track=True, add_pre=False):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout = dropout

        self.add_pre = add_pre

        if add_pre:
            self.preprocess = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.BatchNorm1d(n_hidden, track_running_stats=track),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            )

        if arch is None:
            self.space = [space] * n_layers
        else:
            self.space = [[a] for a in arch]

        self.convs = torch.nn.ModuleList()

        for i in range(n_layers):
            in_feat = n_input if (i == 0 and not add_pre) else n_hidden
            out_feat = n_hidden if i < n_layers - 1 else n_output
            if i < n_layers - 1:
                self.convs.append(Conv(self.space[i], in_feat, out_feat, pos=2, act=F.relu, dropout=dropout, bn=True, res=False, track=track))
            else:
                self.convs.append(Conv(self.space[i], in_feat, out_feat, pos=0))

    def forward(self, data, arch):
        x, edge_index = data.x, data.edge_index
        if self.add_pre:
            x = self.preprocess(x)
        for i, (conv, a) in enumerate(zip(self.convs, arch)):
            x = conv(x, edge_index, a)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.add_pre:
            self.preprocess[0].reset_parameters()
            self.preprocess[1].reset_parameters()

if __name__ == '__main__':
    from itertools import product
    archs = [list(a) for a in product(*[COAUTHOR for _ in range(2)])]
    import pickle
    pickle.dump(archs, open('models/coauthor/archs.pkl', 'wb'))
