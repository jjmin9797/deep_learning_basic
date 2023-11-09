import torch

def reparameterization(mu, logvar) :
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std)
    return mu + eps * std