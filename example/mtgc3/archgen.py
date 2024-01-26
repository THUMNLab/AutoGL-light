import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StructureMask(nn.Module):
    def __init__(self, args, hidden_size, n_chunks, stru_size):
        super().__init__()
        self.args = args
        self.stru_size = stru_size
        self.qus = nn.Linear(hidden_size, n_chunks * stru_size)
        self.qvs = nn.Linear(hidden_size, n_chunks * stru_size)

    def forward(self, x, e, ratio):
        # e: e * 2
        eu, ev = e[0,:], e[1,:]
        eu = x.index_select(0, eu)
        ev = x.index_select(0, ev)
        # get efeat
        # e * d
        ku = self.qus(eu) # e * (chunk * p_s)
        kv = self.qvs(ev)
        ks = ku * kv      # e * (chunk * p_s)
        sws = ks.hsplit(self.stru_size)  # chunk * [e * p_s]
        stru_weights = [i.sum(dim = 1).sigmoid() * (1-ratio) * 0.5 + 1 for i in sws]
        return stru_weights

class AGLayer(nn.Module):
    def __init__(self, args, num_op, n_chunks):
        super().__init__()
        self.args = args
        self.op_emb = nn.Embedding(num_op, args.proto_dim) # op * p_d
        self.task_emb = nn.Embedding(n_chunks, args.proto_dim) # chunk * p_d

    def forward(self):
        o = self.op_emb.weight
        o = o / o.norm(2, dim = -1, keepdim = True) # op * p_d
        cosloss = (o @ o.t()).sum()

        t = self.task_emb.weight
        t = t / t.norm(2, dim = -1, keepdim = True) # chunk * p_d

        alpha = t @ o.t()   # chunk * op
        alpha = alpha / self.args.temperature
        alpha = F.softmax(alpha, dim = 1) # chunk * op
        #alpha = alpha * (alpha > 1/6)
        #alpha = alpha / alpha.sum(dim = 1, keepdim = True)

        #device = alpha.device
        #alpha = torch.Tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #alpha = alpha.tile((self.args.n_chunks, 1)).to(device)

        return alpha, cosloss

class AG(nn.Module):
    def __init__(self, args, num_op, n_chunks):
        super().__init__()
        self.args = args
        self.n_chunks = n_chunks
        self.alphas = nn.ParameterList()
        self.dm = DiagMask(args)
        self.set = 'no'
        for i in range(args.num_layers):
            self.alphas.append(torch.ones(num_op, n_chunks, requires_grad = True))
            #self.alphas.append(torch.ones(num_op, requires_grad = True))

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'set':
            self.dm.set = value

    def forward(self):
        alphas = []
        for alpha in self.alphas:
            alpha = alpha * self.args.alpha_temp
            alpha = F.softmax(alpha, dim = 0) # op * n_chunk

            # same alpha
            #alpha = alpha.tile(self.n_chunks, 1).T

            # fix alpha
            """alpha = torch.zeros(7, 12)
            for i in range(12):
                #alpha[i//2][i] = 1
                alpha[1][i] = 1
            #alpha = torch.ones(8, 8) * 0.125"""

            alpha = alpha.to(self.alphas[0].device)
            alphas.append(alpha)

        if self.set == 'train':
            #print(self.set)
            print('alpha')
            print(alphas[0])
            print(alphas[1])
            print(alphas[-1])
            print('end alpha')
        elif self.set == "paint":
            f = open("info/alpha_fake.txt", "w")
            for alpha in alphas:
                f.write(str(alpha.tolist()) + '\n')
            f.close()

        masks = self.dm()

        return alphas, masks

class DiagMask(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_chunks = args.n_chunks
        chunk_size = args.hidden_size // args.n_chunks
        self.args = args
        self.masks = nn.ParameterList()
        self.set = 'no'
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.hidden_size = n_chunks * chunk_size
        self.neye = torch.ones(n_chunks, n_chunks) - torch.eye(n_chunks)
        self.eye = torch.eye(n_chunks)
        for i in range(args.num_layers):
            #self.masks.append((1 / n_chunks) * torch.ones(n_chunks, n_chunks, requires_grad = True))
            self.masks.append(torch.zeros(n_chunks, n_chunks, requires_grad = True))
            #self.masks.append(math.atanh(0.5) * torch.ones(n_chunks, n_chunks, requires_grad = True))
        
    def forward(self):
        masks = []
        self.neye = self.neye.to(self.masks[0].device)
        self.eye = self.eye.to(self.masks[0].device)
        if self.set == "paint":
            f = open("info/mask_fake.txt", "w")
        for mask in self.masks:
            #m = torch.sigmoid(mask * self.args.mask_temp)
            m = torch.tanh(mask * self.args.mask_temp)
            m = m * self.neye
            m = m + self.eye

            if self.set == 'train':
                #print(self.set)
                print('mask')
                print(m)
                print('end mask')
            elif self.set == "paint":
                f.write(str(m.abs().tolist()) + '\n')

            m = m.tile(self.chunk_size, self.chunk_size).reshape(self.chunk_size, self.n_chunks, self.chunk_size, self.n_chunks).permute(1, 0, 3, 2).reshape(self.hidden_size, -1)
            masks.append(m)
        if self.set == "paint":
            f.close()

        return masks

def Collect_main_para_old(m):
    m.main_paras = torch.nn.ParameterList()
    for p in m.main_para():
        mp = p.clone()
        m.main_paras.append(mp)
        #p.requires_grad_(False)
        #p.is_leaf = False

def Apply_mask_old(m, mask):
    for p, mp in zip(m.main_para(), m.main_paras):
        p.data = mp * mask

def Collect_main_para(m, idim, odim):
    m.main_paras = torch.nn.ParameterList()
    m.main_para_name = []
    for k in m._parameters:
        if 'weight' in k and getattr(m, k).size()[-2:] == torch.Size([idim, odim]):
            mp = getattr(m, k).clone()
            m.main_paras.append(mp)
            m.main_para_name.append(k)
    for k in m.main_para_name:
        del(m._parameters[k])
    
    for mk in m._modules:
        subm = getattr(m, mk)
        for k in subm._parameters:
            if 'weight' in k and getattr(subm, k).size()[-2:] == torch.Size([idim, odim]):
                mp = getattr(subm, k).clone()
                m.main_paras.append(mp)
                m.main_para_name.append((mk, k))
    for k in m.main_para_name:
        if type(k) == tuple and hasattr(getattr(m, k[0]), k[1]):
            del(getattr(m, k[0])._parameters[k[1]])

    #print(m.main_para_name)

def Apply_mask(m, mask):
    for mp, name in zip(m.main_paras, m.main_para_name):
        if type(name) == str:
            setattr(m, name, mp * mask)
            #setattr(m, name, mp)
        else:
            setattr(getattr(m, name[0]), name[1], mp * mask)
            #setattr(getattr(m, name[0]), name[1], mp)
