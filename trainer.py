import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import torchvision.models as pretrained_models
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pretrained VGG16 and get the needed layer
        vgg = pretrained_models.vgg16(pretrained=True).features
        self.vgg_slice = nn.Sequential(*list(vgg.children())[:4]).eval() # Use up to the second convolutional layer
        for param in self.vgg_slice.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Compute feature representations
        x_features = self.vgg_slice(x)
        y_features = self.vgg_slice(y)
        # Compute the Mean Squared Error in feature space
        return F.mse_loss(x_features, y_features)


class Trainer():
    def __init__(self, model, optimizer, dataloader, device):
        self.perceptual_loss = PerceptualLoss()
        self.


def train_epoch(vae, device, dataloader, optimizer, perceptual_loss=None):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Use the MSE loss if perceptual_loss is not provided
    loss_fn = nn.MSELoss(reduction='sum') if perceptual_loss is None else perceptual_loss
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = loss_fn(x, x_hat) + vae.encoder.kl
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulate the batch loss
        train_loss += loss.item()
    # Return the average loss
    return train_loss / len(dataloader.dataset)


def test_epoch(vae, device, dataloader, perceptual_loss=None):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    # Use the MSE loss if perceptual_loss is not provided
    loss_fn = nn.MSELoss(reduction='sum') if perceptual_loss is None else perceptual_loss
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Forward pass through the VAE
            x_hat = vae(x)
            # Compute the loss
            loss = loss_fn(x, x_hat) + vae.encoder.kl
            # Accumulate the batch loss
            val_loss += loss.item()
    # Return the average loss
    return val_loss / len(dataloader.dataset)

