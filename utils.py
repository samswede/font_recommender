import numpy as np
import random
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


def plot_ae_outputs(encoder, decoder, test_dataset, device, n=8, indices=None):
    """
    Function to plot and compare original images from the test dataset and their reconstructions by an autoencoder.

    Parameters:
    - encoder: Trained encoder part of the autoencoder
    - decoder: Trained decoder part of the autoencoder
    - test_dataset: The dataset from which we sample 'n' images to visualize the reconstruction
    - device: The device type used for computation (e.g., 'cuda' or 'cpu')
    - n: The number of sample images to be plotted. Default is 8.
    - indices: Specific indices of images to be plotted. Default is None (images are selected randomly).

    The function plots two rows of images:
    - The first row includes the original images from the test dataset
    - The second row includes the reconstructed images by the autoencoder
    """
    if indices is None:
        # Randomly sample 'n' indices if not provided
        indices = random.sample(range(len(test_dataset)), n)

    plt.figure(figsize=(10,4.5))

    # Set the encoder and decoder in evaluation mode
    encoder.eval()
    decoder.eval()

    for i, idx in enumerate(indices):
        ax = plt.subplot(2, n, i+1)
        img = test_dataset[idx][0].unsqueeze(0).to(device)

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


def visualize_first_layer_filters(vae, num_filters_to_plot=8):
    """ 
    Visualizes the filters in the first convolutional layer of the Variational Autoencoder (VAE).
    Each filter is plotted as an image, with intensity values representing the weights of the filter.
    
    Parameters:
    ----------
    vae : VariationalAutoencoder
        The trained Variational Autoencoder model.
        
    num_filters_to_plot : int, optional
        The number of filters to display from the first layer. The default value is 8, which corresponds to all filters in the first layer.
    """
    # Make sure that the first layer of the encoder part of the VAE is a Conv2d layer
    assert isinstance(vae.encoder.encoder_conv1[0], nn.Conv2d)
    
    # Get the filters from the first layer of the encoder
    filters = vae.encoder.encoder_conv1[0].weight.data.cpu().numpy()

    # Calculate the number of rows needed
    rows = num_filters_to_plot // 4

    # Create subplots
    fig, axs = plt.subplots(rows, 4, figsize=(10, rows*2.5))

    # Flatten the axes
    axs = axs.flatten()
    
    # Go over each subplot and plot the corresponding filter
    for i in range(num_filters_to_plot):
        axs[i].imshow(filters[i, 0], cmap='gray')  # We only display the first channel
        axs[i].axis('off')
        axs[i].set_title(f'Filter {i+1}')
        
    # Delete unused subplots
    for i in range(num_filters_to_plot, len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.show()

