import torch
from torch import nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        ### Convolutions 1
        #Input: 128x128x1
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #Input: 128x128x8
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
            #Input: 64x64x16
        )

        ### Convolutions 2
        #Input: 64x64x16
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #Input: 64x64x128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
            #Input: 32x32x256

        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        # Input:  256 * 32 * 32 = 262'144


        self.linear1 = nn.Linear(256*32*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)
        
        ### Probablistic section
        self.N = torch.distributions.Normal(0, 1)

        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        #print(x.shape)
        #x = x.to(device)
        x = self.encoder_conv1(x)
        #print("shape after encoder_conv1:")
        #print(x.shape)
        x = self.encoder_conv2(x)
        #print("shape after encoder_conv2:")
        #print(x.shape)
        x = self.flatten(x)
        #print("shape after flatten:")
        #print(x.shape)

        x = F.relu(self.linear1(x))
        #print("shape after linear1:")
        #print(x.shape)

        # create mean latent vector
        mu = self.linear2(x)
        # create std latent vector
        sigma = torch.exp(self.linear3(x))

        # sample latent vector z from mean and std latent vectors
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        # return the sampled latent vector z
        return z
    

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 256*32*32),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 32, 32))

        ### Convolutional section 1
        # H_out =(H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        self.decoder_conv1 = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=16, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(16)
        )

        ### Convolutional section 2
        self.decoder_conv2 = nn.Sequential(
            #Input:  64x64x16
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #Input:  128x128x8
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0)
            #nn.BatchNorm2d(1)
            #nn.ReLU(True) #may have to remove. Do not see in example...
            #Output:  128x128x1
        )

        
    def forward(self, x):

        # Apply linear layers
        x = self.decoder_lin(x)
        #print("shape after decoder_lin:")
        #print(x.shape)
        # Unflatten
        x = self.unflatten(x)
        #print("shape after unflatten:")
        #print(x.shape)
        # Apply transposed convolutions
        x = self.decoder_conv1(x)
        #print("shape after decoder_conv1:")
        #print(x.shape)
        x = self.decoder_conv2(x)
        #print("shape after decoder_conv2:")
        #print(x.shape)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x
    

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        #x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
