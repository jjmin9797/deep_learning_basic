import torch.nn as nn
import torch
import torch.nn.functional as F

import util
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()

        #First hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        #Second hidden layer
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        #Output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self,x):
        x = self.fc2(self.fc1(x))
        mu = F.relu(self.mu(x))
        logvar = F.relu(self.logvar(x))

        z = util.reparameterization(mu, logvar)
        return z, mu, logvar

