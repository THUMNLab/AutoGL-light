import torch
import torch.nn as nn
import torch.nn.functional as F

class AGLayer(nn.Module):
    def __init__(self, args, num_op):
        super().__init__()
        self.args = args
        self.op_emb = nn.Embedding(num_op, args.graph_dim) # op * g_d

    def forward(self, g):
        # g: graph * g_d
        o = self.op_emb.weight
        o = o / o.norm(2, dim = -1, keepdim = True)
        cosloss = (o @ o.t()).sum()
        alpha = g @ o.t()
        alpha = alpha / self.args.temperature
        alpha = F.softmax(alpha, dim = 1) # graph * op
        
        # GDAS
        '''index = alpha.max(-1, keepdim=True)[1]
        one_h = torch.zeros_like(alpha).scatter_(-1, index, 1.0)
        alpha = one_h - alpha.detach() + alpha'''

        alpha = alpha * (alpha > 1/6)
        alpha = alpha / alpha.sum(dim = 1, keepdim = True)
        return alpha, cosloss

class AG(nn.Module):
    def __init__(self, args, num_op, num_pool):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList()
        self.set = 'train'
        for i in range(args.num_layers + 1):
            self.layers.append(AGLayer(args, num_op))

    def forward(self, g):
        # g: graph * g_d
        alpha_all = []
        cosloss = torch.zeros(1).to(self.layers[0].op_emb.weight.device)
         
        '''if self.set == "train":
            print('graph')
            print(g[:8])
            gg = g[:8] / g[:8].norm(2, dim = -1, keepdim=True)
            gg = gg @ gg.t()
            print(gg)
            print('end graph')

            print('op')
            op = self.layers[0].op_emb.weight
            oo = op / op.norm(2, dim = -1, keepdim = True)
            oo = oo @ oo.t()
            print(op)
            print(oo)
            print('end op')'''
            
        for i in range(self.args.num_layers):
            alpha, closs = self.layers[i](g)
            cosloss = cosloss + closs
            alpha_all.append(alpha)
        
        '''if self.set == 'train':
            #print(self.set)
            print('alpha')
            print(alpha_all[0][:8])
            print(alpha_all[1][:8])
            print(alpha_all[2][:8])
            print('end alpha')'''

        #alpha_all = [i.detach() for i in alpha_all]
        
        return alpha_all, cosloss
