import os
import numpy as np
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets,transforms
from torchvision.utils import save_image
import Decoder
import Encoder
image_path = './images'
channels = 1

n_epochs = 100
batch_size = 128
lr = 0.0001
b1 = 0.5
b2 = 0.999

img_size = 28
h_dim = 512
latent_dim = 20

device = torch.device("cuda")

transform = transforms.Compose([
    transforms.ToTensor()
])

train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_dataloader = DataLoader(
    train,
    batch_size = batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test,
    batch_size=batch_size,
    shuffle=False
)

encoder = Encoder.Encoder(img_size** 2, h_dim, latent_dim).to(device)
decoder = Decoder.Decoder(img_size** 2, h_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
)

total_loss = []
for epoch in range(n_epochs):
    train_loss = 0
    for i, (x,_) in enumerate(train_dataloader):
        x = x.view(-1,img_size**2)
        x = x.to(device)
        z, mu, logvar = encoder(x)
        x_reconst = decoder(z)

        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        loss = reconst_loss + kl_div
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_dataloader)}], Reconst Loss : {reconst_loss.item():.4f}, KL Div: {kl_div.item():.4f}')
        print(f'===> Epoch: {epoch+1} Average Train Loss: {train_loss/len(train_dataloader.dataset):.4f} ')
    
    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_dataloader):
            # forward
            x = x.view(-1, img_size**2)
            x = x.to(device)
            z, mu, logvar = encoder(x)
            x_reconst = decoder(z)

            # compute reconstruction loss and KL divergence
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

            loss = reconst_loss + kl_div
            test_loss += loss.item()
            if i==0:
                x_concat = torch.cat([x.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
                # batch size 개수만큼의 이미지 쌍(input x, reconstructed x)이 저장됨
                save_image(x_concat, os.path.join(image_path,f'reconst-epoch{epoch+1}.png'))
        print(f'===> Epoch: {epoch+1} Average Test Loss: {test_loss/len(test_dataloader.dataset):.4f} ')
        z = torch.randn(batch_size, latent_dim).to(device) # N(0, 1)에서 z 샘플링
        sampled_images = decoder(z)
        save_image(sampled_images.view(-1, 1, 28, 28), os.path.join(image_path,f'sampled-epoch{epoch+1}.png'))
            
