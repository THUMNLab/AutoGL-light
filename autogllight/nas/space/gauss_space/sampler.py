import math
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import scipy
from .supernet.ops import uniform_sample

class BaseArchSampler(object):
    def __init__(self, space, num_layers=5):
        self.space = space
        self.num_layers = num_layers

    def sample(self):
        return uniform_sample(self.num_layers, self.space), 1.

    def fit(self, arch2score):
        return

    def train(self):
        return

    def eval(self):
        return

    def log(self, logfunc):
        return

    def save(self, folder, epoch):
        return

    def samples(self, num_sample):
        return [self.sample() for _ in range(num_sample)]

    def restart(self):
        return

class ReSampler(BaseArchSampler):
    def __init__(self, space, num_layers=5, sample_ratio=2, target="loss"):
        super().__init__(space, num_layers)
        self.sample_ratio = sample_ratio
        self.arch2score = None
        self.target = target

    def samples(self, num_samples):
        sample_time = int(num_samples * self.sample_ratio)
        # first, random sample several architectures
        archs = [uniform_sample(self.num_layers, self.space) for _ in range(sample_time)]
        # define the distribution from these archs
        losses, acc, grad = self.arch2score(archs, gradient = (self.target=="grad"))
        if self.target == "acc":
            target = acc
        elif self.target == "loss":
            target = losses
        elif self.target == "grad":
            target = grad
        elif self.target == "nloss":
            target = [-l for l in losses]
        scores = [x / sum(target) for x in target]
        selected = random.choices(list(range(sample_time)), weights=scores, k=num_samples)
        return [[archs[i], 1 / sample_time / scores[i]] for i in selected]


class RandomSampler(BaseArchSampler):
    ...

class ExpSampler(BaseArchSampler):
    def __init__(self, space, num_layers=5, num_samples=100, temprature=1.0, target="acc"):
        super().__init__(space, num_layers)
        self.num_of_ops = len(self.space)
        self.prob = np.ones((self.num_layers, self.num_of_ops)) / self.num_of_ops
        self.num_samples = num_samples
        self.temprature = temprature
        self.target = target

    def sample(self):
        arch = [None for _ in range(self.num_layers)]
        ratio = 1.0
        for layer in range(self.num_layers):
            idx = np.random.choice(self.num_of_ops, p=self.prob[layer])
            arch[layer] = self.space[idx]
            ratio *= 1.0 / self.prob[layer][idx] / self.num_of_ops
        return arch, ratio

    def evaluate(self, archs):
        scores = []
        for arch in archs:
            p = 0.0
            for i, a in enumerate(arch):
                idx = self.space.index(a)
                p += np.log(self.prob[i][idx] + 1e-3)
            scores.append(p)
        return scores

    def fit(self, arch2score):
        archs = [uniform_sample(self.num_layers, self.space) for _ in range(self.num_samples)]
        losses, acc, grad = arch2score(archs, self.target=="grad")
        if self.target == "acc":
            target = acc
        elif self.target == "loss":
            target = losses
        elif self.target == "grad":
            target = grad
        elif self.target == "nloss":
            target = [-l for l in losses]
        bin = np.zeros((self.num_layers, self.num_of_ops))
        count = np.zeros((self.num_layers, self.num_of_ops))
        for arch, value in zip(archs, target):
            for layer in range(self.num_layers):
                bin[layer][self.space.index(arch[layer])] += value
                count[layer][self.space.index(arch[layer])] += 1

        bin = bin / count
        self._cal_prob(bin)
        return

    def _cal_prob(self, bin):
        self.prob = scipy.special.softmax(bin / self.temprature, axis=1)

    def log(self, logfunc):
        logfunc(str(self.prob))

    def save(self, folder, epoch):
        import pickle
        pickle.dump(self.prob, open(os.path.join(folder, f"{epoch}-prob.pkl"), "wb"))


class BayesSampler(ExpSampler):
    def _cal_prob(self, bin):
        self.prob = bin / bin.sum(axis=1, keepdims=True)

def select(array, index):
    # print("check device")
    # print(array.device, index.device)
    eye_on_gpu = torch.eye(array.size(1), device=array.device)
    return array[eye_on_gpu[index].bool()]

class DifferentiableSampler(BaseArchSampler):
    def __init__(self, space, num_layers=5, device='cuda', repeat=10, epoch=5, lr=1e-3):
        super().__init__(space, num_layers=num_layers)
        self.param = torch.nn.Parameter(torch.ones(self.num_layers, len(self.space)))
        self.optimizer = torch.optim.Adam([self.param], lr=lr)
        self.device = device
        self.repeat = repeat
        self.epoch = epoch
        self.lr = lr

    def restart(self):
        self.param = torch.nn.Parameter(torch.ones(self.num_layers, len(self.space)))
        self.optimizer = torch.optim.Adam([self.param], lr=self.lr)

    @property
    def prob(self):
        with torch.no_grad():
            return F.softmax(self.param, dim=1).cpu().numpy()

    def evaluate(self, archs):
        scores = []
        for arch in archs:
            p = 0.0
            for i, a in enumerate(arch):
                idx = self.space.index(a)
                p += np.log(self.prob[i][idx] + 1e-3)
            scores.append(p)
        return scores

    def log(self, log):
        log(str(self.prob))

    def save(self, folder, epoch):
        torch.save(self.param, os.path.join(folder, f"{epoch}.pth"))

    @torch.no_grad()
    def sample(self):
        # sample from gumbel-softmax
        parameters = self.param.to(self.device)
        sampled = F.gumbel_softmax(parameters, dim=-1)
        arch_list = sampled.argmax(dim=1)
        prob = F.softmax(parameters, dim=-1)
        ratio = 1.0
        for a, p in zip(arch_list, prob):
            ratio *= (1 / len(self.space) / p[a]).item()
        return [self.space[i] for i in arch_list.tolist()], ratio

    def fit(self, model, data, mask):
        # sample archs and back-prop through gumbel
        model.eval()
        for _ in range(self.epoch):
            self.optimizer.zero_grad()
            for _ in range(self.repeat):
                parameters = self.param.to(self.device)
                sampled = F.gumbel_softmax(parameters, hard=True, dim=-1)
                multiplier = sampled.max(dim=1)[0]
                arch = [self.space[i] for i in sampled.argmax(dim=1).tolist()]
                out = model(data.x, data.adj_t, arch, multiplier)
                loss = F.nll_loss(out[mask], data.y.squeeze(1)[mask]) / self.repeat
                loss.backward()
            self.optimizer.step()


class RNNAgent(torch.nn.Module):
    def __init__(self, n_words, n_lengths, n_dims=100, T=1.):
        # n_words will be <begin>
        super().__init__()
        self.embedding = torch.nn.Embedding(n_words + 1, n_dims)
        self.idx_embedding = torch.nn.Embedding(n_lengths, n_dims)
        self.cell = torch.nn.GRUCell(n_dims, n_dims)
        self.out = torch.nn.Linear(n_dims, n_words, bias=False)
        self.out.weight = self.embedding.weight
        self.embedding.weight.data.mul_(0.01)
        self.n_lengths = n_lengths
        self.n_words = n_words
        self.T = T

    def sample(self, n_samples):
        # rolling up
        inp = (torch.ones(n_samples, device=next(self.parameters()).device) * self.n_words).long()
        idx = torch.ones(n_samples, device=next(self.parameters()).device)
        x = self.embedding(inp) + self.idx_embedding((idx * 0).long())
        hidden = None
        sampled_archs = []
        log_probs = []
        probs = []
        entropy = []
        for i in range(self.n_lengths):
            hidden = self.cell(x, hidden)
            # out = F.linear(hidden, self.embedding.weight[:-1]) / 10.
            out = self.out(hidden)[:,:-1] / self.T
            prob = F.softmax(out, dim=1)
            log_prob = F.log_softmax(out, dim=1)
            sampled = torch.multinomial(prob, 1)
            log_probs.append(select(log_prob, sampled[:,0]))
            probs.append(select(prob, sampled[:,0]))
            sampled_archs.append(sampled)
            entropy.append(- (log_prob * prob).sum(dim=1))
            if i < self.n_lengths - 1:
                x = self.embedding(sampled[:,0]) + self.idx_embedding((idx * (i + 1)).long())

        sampled_archs = torch.cat(sampled_archs, dim=1)
        return sampled_archs.tolist(), log_probs, probs, sum(entropy)

    @torch.no_grad()
    def evaluate(self, archs):
        self.eval()
        n_samples = len(archs)
        inp = (torch.ones(n_samples, device=next(self.parameters()).device) * self.n_words).long()
        idx = torch.ones(n_samples, device=next(self.parameters()).device)
        x = self.embedding(inp) + self.idx_embedding((idx * 0).long())
        hidden = None
        sampled_archs = []
        log_probs = []
        probs = []
        for i in range(self.n_lengths):
            hidden = self.cell(x, hidden)
            out = self.out(hidden)[:,:-1]
            log_prob = F.log_softmax(out, dim=1)
            sampled = torch.tensor([a[i] for a in archs]).long().to(log_prob.device)
            log_probs.append(select(log_prob, sampled))
            if i < self.n_lengths - 1:
                x = self.embedding(sampled) + self.idx_embedding((idx * (i + 1)).long())

        log_probs = sum(log_probs).tolist()
        return log_probs


class RLSampler(BaseArchSampler):
    def __init__(self, space, num_layers=5, device='cuda', target="acc", epochs=50, iter=2, lr=1e-3, T=1., entropy=0.):
        super().__init__(space, num_layers=num_layers)
        self.num_of_ops = len(space)
        self.sampler = RNNAgent(self.num_of_ops, num_layers, T=T).to(device)
        self.optimizer = torch.optim.Adam(self.sampler.parameters(), lr=lr)
        self.lr = lr
        self.target = target
        self.device = device
        self.epoch = epochs
        self.iter = iter
        self.T = T
        self.entropy = entropy
        self.fitted = False

    def restart(self):
        self.sampler = RNNAgent(self.num_of_ops, self.num_layers, T=self.T).to(self.device)
        self.optimizer = torch.optim.Adam(self.sampler.parameters(), lr=self.lr)
        self.fitted = False

    def train(self):
        self.sampler.train()

    def eval(self):
        self.sampler.eval()

    @torch.no_grad()
    def sample(self):
        if not self.fitted:
            return uniform_sample(self.num_layers, self.space), 1.
        self.sampler.eval()
        arch, log_p, p, _ = self.sampler.sample(1)
        ratio = 1.0
        for p_e in p:
            ratio *= 1 / self.num_of_ops / p_e.item()
        return [self.space[i] for i in arch[0]], ratio

    def fit(self, arch2score):
        self.fitted = True
        self.sampler.train()
        baseline = None
        for _ in range(self.epoch):
            self.sampler.train()
            self.optimizer.zero_grad()
            archs, log_p, _, entropy = self.sampler.sample(self.iter)
            _, accs = arch2score([[self.space[i] for i in arch] for arch in archs])
            target = accs
            if baseline is None: baseline = np.mean(target)
            else:
                baseline = baseline * 0.9 + np.mean(target) * 0.1
            reward = torch.tensor([l - baseline for l in target]).to(log_p[0].device)
            loss_REINFORCE = - sum([(reward * lp).mean() for lp in log_p])
            if self.entropy > 0:
                loss_REINFORCE -= entropy.mean() * self.entropy
            loss_REINFORCE.backward()
            self.optimizer.step()

    def save(self, folder, epoch):
        torch.save(self.sampler.state_dict(), os.path.join(folder, f"{epoch}-sampler.pth"))

    def evaluate(self, archs):
        arch2int = [[self.space.index(a) for a in arch] for arch in archs]
        return self.sampler.evaluate(arch2int)
