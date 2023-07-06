import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import torchvision.models as pretrained_models
import torch.nn.functional as F

"""TO DO:
    - FIX combine_all_vectors_and_labels() to do what it says in the comment.
        This is critical for it to work, and to load the correct images from the folder using the backend later.
"""

def load_data_dict(file_path):
    with open(file_path, 'rb') as handle:
        loaded_dict = pickle.load(handle)
    return loaded_dict

def save_data_dict(file_name, map_labels_to_indices):
    # assuming map_labels_to_indices is your dictionary
    with open(f'{file_name}.pickle', 'wb') as handle:
        pickle.dump(map_labels_to_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass

def save_npz(array, file_name):
    np.savez(file_name, array=array)

def load_npz(file_path):
    with np.load(file_path) as data:
        numpy_array = data['array']

    return numpy_array

def combine_all_vectors_and_labels(path):
    """
    Creates a dictionary to both keep track of all font names and
    allow us to translate from font names to index in the training/testing sets
    """
    file_list = [file for file in os.listdir(path) if file.endswith('.png')]
    
    # Preallocate a list of arrays
    arrays_list = []
    
    font_name_to_index = {}
    
    for index, file in enumerate(file_list):

        array = np.load(os.path.join(path, file))
        arrays_list.append(array)

        # Extract the label name from the file name by removing '.npy' and add it to the dictionary
        font_name = os.path.splitext(file)[0].replace('diffusion_profile_', '')
        font_name_to_index[font_name] = index
    
    return font_name_to_index


def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size = 6, shuffle = True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow = 3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))


def plot_ae_outputs(encoder, decoder, test_dataset, device, n=8):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = test_dataset[(3*i)**2+i][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img  = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Original images')
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Reconstructed images')
    plt.show()   


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

