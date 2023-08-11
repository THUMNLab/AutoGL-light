import torch
from torch import nn, optim
from torch.nn import functional as F

class AE(nn.Module):
    ''' Autoencoder for dimensional reduction'''
    def __init__(self,dim, hidden_dim = 128, hidden_length = 0):
        super(AE, self).__init__()
        self.dim = dim
        self.hidden_length = hidden_length
        self.fc1 = nn.Linear(dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, dim)

        self.layers = nn.ModuleList()
        for _ in range(hidden_length):
            layer = nn.Linear( hidden_dim * 4, hidden_dim * 4)
            self.layers.append(layer)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        for i in range(self.hidden_length):
            z = F.relu(self.layers[i](z))
        return self.decode(z), z

class VAE(nn.Module):
    ''' Variational Autoencoder for dimensional reduction'''
    def __init__(self,dim, hidden_dim = 20, hidden_length = 0):
        super(VAE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden_dim * hidden_dim)
        self.fc21 = nn.Linear(hidden_dim * hidden_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim * hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        self.fc4 = nn.Linear(hidden_dim * hidden_dim, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
