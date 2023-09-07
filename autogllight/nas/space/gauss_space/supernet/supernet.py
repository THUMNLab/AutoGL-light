import torch
import torch.nn.functional as F
from .conv import Conv

class Supernet(torch.nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden,
        n_classes,
        n_layers,
        dropout=0.5,
        act=torch.nn.ReLU(),
        track=True,
        space=['gin', 'gatv2', 'gcn', 'sage', 'graph', 'linear'],
        arch=None,
        add_pre=False
    ) -> None:
        super().__init__()
        if arch is not None:
            self.space = [[a] for a in arch]
        else:
            self.space = [space] * n_layers
        self.act = act
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        self.add_pre = add_pre
        if add_pre:
            self.preprocess = torch.nn.Sequential(
                torch.nn.Linear(n_features, n_hidden),
                torch.nn.BatchNorm1d(n_hidden, track_running_stats=track),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            )

        for i in range(n_layers - 1):
            in_c = n_features if (i == 0 and not add_pre) else n_hidden
            out_c = n_hidden
            self.convs.append(Conv(self.space[i], in_c, out_c, 2, act, dropout, True, False, track))
        self.convs.append(Conv(self.space[n_layers - 1], n_hidden, n_classes, 0))

    def forward(self, x, adj, arch=None):
        if arch is None: arch = [x[0] for x in self.space]
        if self.add_pre:
            x = self.preprocess(x)
        for i, (conv, a) in enumerate(zip(self.convs, arch)):
            x = conv(x, adj, a)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.add_pre:
            self.preprocess[0].reset_parameters()
            self.preprocess[1].reset_parameters()
