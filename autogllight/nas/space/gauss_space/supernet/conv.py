import torch
from .ops import op, op_dgl
import torch.nn.functional as F
from torch.nn import Identity

class ConvDGL(torch.nn.Module):
    def __init__(self, space, input_dim, output_dim, pos=0, act=Identity(), dropout=0.0, bn=False, res=False, track=False, edge_feat=16, **kwargs) -> None:
        super().__init__()
        self.pos = pos
        self.act = act
        self.bn = bn
        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = Identity()
        self.space = space
        self.res = res

        # span a space
        self.core = torch.nn.ModuleDict({name: op_dgl(name, input_dim, output_dim, edge_feat, **kwargs) for name in space})
        
        self.bns = torch.nn.ModuleDict({
            name: (
                torch.nn.BatchNorm1d(input_dim if self.pos == 1 else output_dim, track_running_stats=track) 
                if self.bn else Identity()
            ) 
        for name in space})

    def forward(self, x, edge_index, key):
        x_origin = x
        if self.pos == 1:
            # norm - act - drop - conv - (possible) res
            if isinstance(x, tuple):
                x = (self.bns[key](x[0]), self.bns[key](x[1]))
                x = (self.act(x[0]), self.act(x[1]))
                x = (self.dropout(x[0]), self.dropout(x[1]))
            else:
                x = self.bns[key](x)
                x = self.act(x)
                x = self.dropout(x)
        
        x = self.core[key](x, edge_index)

        if self.pos == 2:
            # conv - norm - act - drop - res
            x = self.bns[key](x)
            x = self.act(x)
            x = self.dropout(x)
        
        if self.res:
            if isinstance(x_origin, tuple): x = x + x_origin[1]
            else: x = x + x_origin
        
        return x

    def reset_parameters(self):
        for key in self.core:
            self.core[key].reset_parameters()
            if hasattr(self.bns[key], "reset_parameters"):
                self.bns[key].reset_parameters()


class Conv(torch.nn.Module):
    def __init__(self, space, input_dim, output_dim, pos=0, act=Identity(), dropout=0.0, bn=False, res=False, track=False, **kwargs) -> None:
        super().__init__()
        self.pos = pos
        self.act = act
        self.bn = bn
        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = Identity()
        self.space = space
        self.res = res

        # span a space
        self.core = torch.nn.ModuleDict({name: op(name, input_dim, output_dim, **kwargs) for name in space})
        
        self.bns = torch.nn.ModuleDict({
            name: (
                torch.nn.BatchNorm1d(input_dim if self.pos == 1 else output_dim, track_running_stats=track) 
                if self.bn else Identity()
            ) 
        for name in space})

    def forward(self, x, edge_index, key):
        x_origin = x
        if self.pos == 1:
            # norm - act - drop - conv - (possible) res
            if isinstance(x, tuple):
                x = (self.bns[key](x[0]), self.bns[key](x[1]))
                x = (self.act(x[0]), self.act(x[1]))
                x = (self.dropout(x[0]), self.dropout(x[1]))
            else:
                x = self.bns[key](x)
                x = self.act(x)
                x = self.dropout(x)
        
        x = self.core[key](x, edge_index)

        if self.pos == 2:
            # conv - norm - act - drop - res
            x = self.bns[key](x)
            x = self.act(x)
            x = self.dropout(x)
        
        if self.res:
            if isinstance(x_origin, tuple): x = x + x_origin[1]
            else: x = x + x_origin
        
        return x

    def reset_parameters(self):
        for key in self.core:
            self.core[key].reset_parameters()
            if hasattr(self.bns[key], "reset_parameters"):
                self.bns[key].reset_parameters()
