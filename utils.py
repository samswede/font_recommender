import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
#import torchvision.models as pretrained_models
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

def visualize_first_layer_filter_outputs(vae, test_dataset, device, num_filters_to_plot=8):
    """ 
    Visualizes the output of applying each filter in the first convolutional layer of the Variational Autoencoder (VAE) to a randomly sampled image from the test dataset.
    
    Parameters
    ----------
    vae : VariationalAutoencoder
        The trained Variational Autoencoder model.
        
    test_dataset : Dataset
        The test dataset to sample the image from.
        
    device : Device
        The device (CPU or GPU) where the computations will take place.
        
    num_filters_to_plot : int, optional
        The number of filter outputs to display. The default value is 8, which corresponds to all filters in the first layer.
    """
    # Make sure that the first layer of the encoder part of the VAE is a Conv2d layer
    assert isinstance(vae.encoder.encoder_conv1[0], nn.Conv2d)

    # Randomly sample an image from the test dataset
    img_index = np.random.randint(len(test_dataset))
    img = test_dataset[img_index][0].unsqueeze(0).to(device)
    
    # Get the first Conv2d layer
    first_layer = vae.encoder.encoder_conv1[0]

    # Apply the filters to the image and get the output
    output = first_layer(img)[0].detach().cpu().numpy()

    # Calculate the number of rows needed
    rows = num_filters_to_plot // 4

    # Create subplots
    fig, axs = plt.subplots(rows, 4, figsize=(10, rows*2.5))

    # Flatten the axes
    axs = axs.flatten()
    
    # Go over each subplot and plot the corresponding filter output
    for i in range(num_filters_to_plot):
        axs[i].imshow(output[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Filter {i+1} Output')
        
    # Delete unused subplots
    for i in range(num_filters_to_plot, len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.show()

def visualize_deeper_layer_filter_outputs(vae, test_dataset, device, layer_index, num_filters_to_plot=16):
    """ 
    Visualizes the output of applying each filter in the specified convolutional layer of the Variational Autoencoder (VAE) to a randomly sampled image from the test dataset.
    
    Parameters
    ----------
    vae : VariationalAutoencoder
        The trained Variational Autoencoder model.
        
    test_dataset : Dataset
        The test dataset to sample the image from.
        
    device : Device
        The device (CPU or GPU) where the computations will take place.
        
    layer_index : int
        The index of the convolutional layer to visualize. Indexing starts from 0. Here are the possible values:
        - 0: The first Conv2d layer in the encoder_conv1 block (out_channels=8).
        - 1: The second Conv2d layer in the encoder_conv1 block (out_channels=16).
        - 2: The first Conv2d layer in the encoder_conv2 block (out_channels=128).
        - 3: The second Conv2d layer in the encoder_conv2 block (out_channels=256).
        Please note that for layers with a large number of filters, it's recommended to keep num_filters_to_plot relatively small.

    num_filters_to_plot : int, optional
        The number of filter outputs to display. The default value is 16. Here are the possible values for each layer:
        - Layer 0: Any value from 1 to 8.
        - Layer 1: Any value from 1 to 16.
        - Layer 2: Any value from 1 to 128.
        - Layer 3: Any value from 1 to 256.
        It's recommended to keep this number relatively small (e.g., 16 or 32), as deeper layers can have a large number of filters (e.g., 256), and visualizing them all at once can be overwhelming.
    """
    # Get all the Conv2d layers in the encoder
    conv_layers = [module for module in vae.encoder.modules() if isinstance(module, nn.Conv2d)]

    # Make sure the layer index is valid
    assert layer_index < len(conv_layers), "Invalid layer index"

    # Randomly sample an image from the test dataset
    img_index = np.random.randint(len(test_dataset))
    img = test_dataset[img_index][0].unsqueeze(0).to(device)

    # Define a function to apply a sequential block to an input tensor
    def apply_sequential_block(block, x):
        for layer in block:
            x = layer(x)
        return x

    # Apply the layers of the encoder one by one until reaching the specified layer
    x = img
    if layer_index < 4:  # The layer is in the encoder_conv1 block
        x = apply_sequential_block(vae.encoder.encoder_conv1[:layer_index+1], x)
    else:  # The layer is in the encoder_conv2 block
        x = apply_sequential_block(vae.encoder.encoder_conv1, x)  # Apply the whole encoder_conv1 block
        x = apply_sequential_block(vae.encoder.encoder_conv2[:layer_index-3], x)

    # Get the output of applying the filters to the image
    output = x[0].detach().cpu().numpy()

    # Calculate the number of rows needed
    rows = num_filters_to_plot // 4

    # Create subplots
    fig, axs = plt.subplots(rows, 4, figsize=(10, rows*2.5))

    # Flatten the axes
    axs = axs.flatten()

    # Go over each subplot and plot the corresponding filter output
    for i in range(num_filters_to_plot):
        axs[i].imshow(output[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Filter {i+1} Output')

    # Delete unused subplots
    for i in range(num_filters_to_plot, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

def moving_std(x, window_size):
    ret = np.cumsum(x, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    sq = (ret[window_size - 1:] ** 2) / window_size
    mean_sq = sq - ((ret[window_size - 1:] - ret[:-window_size + 1]) ** 2) / window_size ** 2
    return np.sqrt(mean_sq / (window_size - 1))

def plot_loss_progression(train_losses, val_losses, window_size=50):
    """
    This function receives lists of training and validation losses and plots them.
    It applies a moving average filter for smoothing and also calculates the moving standard deviation.
    """
    # Check if there's enough data to apply a moving average
    if len(train_losses) >= window_size:
        # Apply moving average and moving standard deviation
        train_losses_smooth = moving_average(train_losses, window_size)
        val_losses_smooth = moving_average(val_losses, window_size)
        train_losses_std = moving_std(train_losses, window_size)
        val_losses_std = moving_std(val_losses, window_size)
        # Generate x values
        x_values = np.arange(window_size - 1, len(train_losses))
    else:
        train_losses_smooth = train_losses
        val_losses_smooth = val_losses
        train_losses_std = np.zeros(len(train_losses))
        val_losses_std = np.zeros(len(val_losses))
        x_values = np.arange(len(train_losses))

    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, train_losses_smooth, label='Training Loss')
    plt.fill_between(x_values, train_losses_smooth - train_losses_std, train_losses_smooth + train_losses_std, alpha=0.2)
    plt.plot(x_values, val_losses_smooth, label='Validation Loss')
    plt.fill_between(x_values, val_losses_smooth - val_losses_std, val_losses_smooth + val_losses_std, alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Progression of Training and Validation Losses Over Time')
    plt.show()
