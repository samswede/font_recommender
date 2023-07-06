#%%

import matplotlib.pyplot as plt
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
import random # this module will be used to select random samples from a collection
import os # this module will be used just to create directories in the local filesystem
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from OLD_variational_autoencoder import *
from manager import *
from vector_database import *
from dimensionality_reduction import *
from utils import *
from preprocessing import *



#%%
model_path='./models/big_vae_L9_E700.pt'
embeddings_path='./data/embeddings/cleaned_big_L9_E700.csv'

vae = VAEModel(model_path= model_path, 
               embeddings_path= embeddings_path, 
               latent_dims= 9)

#%%
font_1_index = 300
font_2_index = 310
interpolation_fraction = 0.5

font_1_image_numpy, font_2_image_numpy, interpolated_image_numpy = vae.generate_interpolated_images_numpy(font_1_index=font_1_index, font_2_index=font_2_index, interpolation_fraction=interpolation_fraction)
vae.display_images(font_1_image_numpy, font_2_image_numpy, interpolated_image_numpy)

vae.create_interpolation_gif(font_1_index=font_1_index, font_2_index=font_2_index, gif_path=f'interpolation_font_{font_1_index}_to_{font_2_index}.gif')

# %%

def get_similar_fonts(chosen_font_label, distance_metric='euclidean'):
    
    # Translate indication name to index in indication diffusion profiles, to retrieve diffusion profile
    #chosen_font_label = graph_manager.mapping_indication_name_to_label[chosen_indication_name]
    chosen_font_index = dict_font_labels_to_indices[chosen_font_label]
    chosen_font_diffusion_profile = font_embeddings_array[chosen_font_index]

    #====================================
    # Querying Vector Database to return drug candidates
    #====================================
    num_recommendations = 10

    query = chosen_font_diffusion_profile

    font_candidates_indices = font_vector_db.nearest_neighbors(query, distance_metric, num_recommendations)

    font_candidates_labels = [dict_font_indices_to_labels[index] for index in font_candidates_indices]
    #drug_candidates_names = [graph_manager.mapping_drug_label_to_name[i] for i in font_candidates_labels]

    return font_candidates_labels # List


#%%

def create_image_embedding(font_image_file_name, font_images_path, transform_norm, model):

    #font_image_file_name = f'{font_name}_Aa.png'
    font_image_path = f'{font_images_path}{font_image_file_name}'

    print(font_image_path)

    # Read the image file
    image = Image.open(font_image_path)

    # Apply the transformation
    image = transform_norm(image)
    
    # Add an extra dimension for batch (PyTorch models expect a batch dimension)
    image = image.unsqueeze(0)

    # pass tensor image through vae encoder
    embedding_torch = model.encoder(image)

	# convert torch tensor to numpy array
    embedding_numpy = embedding_torch.detach().numpy()

    return embedding_numpy

def combine_all_embeddings_into_array(font_images_path, model, transform_norm):
    file_list = [file for file in os.listdir(font_images_path) if file.endswith('.png')]

    print(file_list)
    # Preallocate a list of arrays, and empty dict
    embeddings_list = []
    font_name_to_index = {}

    for index, font_image_file_name in enumerate(file_list):

        embedding_numpy = create_image_embedding(font_image_file_name, font_images_path, transform_norm, model)
        embeddings_list.append(embedding_numpy)

        # Extract the label name from the file name by removing '.npy' and add it to the dictionary
        font_name = font_image_file_name.replace('_Aa.png', '')
        font_name_to_index[font_name] = index
    
    all_font_embeddings = np.vstack(embeddings_list)

    return font_name_to_index, all_font_embeddings



#%%
""" TO DO:
        - I suspect that i need to split the dataset into train and test folders.
        - Wait actually i think that I need to print the length of the dataset and make sure the indices are correct.

"""

dataset_path = './data/fonts/Aa_improved/'

train_loader, valid_loader, test_loader, train_dataset, test_dataset, transform_norm = prepare_data_loaders(dataset_path)

#%%
all_font_images_path = './data/fonts/all_font_images/'
font_name_to_index, all_font_embeddings = combine_all_embeddings_into_array(font_images_path= all_font_images_path, model= vae.model, transform_norm= transform_norm)

#%%
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
#%%
save_npz(all_font_embeddings, 'all_font_embeddings')

#%%
save_data_dict(file_name='font_name_to_index', map_labels_to_indices=font_name_to_index)

#%%
dictionary_path = './data/embeddings/font_name_to_index.pickle'
dict_font_labels_to_indices= load_data_dict(dictionary_path)
#%%
font_embeddings_path = './data/embeddings/all_font_embeddings.npz'
font_embeddings_array = load_npz(file_path= font_embeddings_path)
#%%

# load font embeddings as pandas df
df = pd.read_csv(embeddings_path)

# drop index column
df = df.drop(df.columns[0], axis=1)

# Convert the DataFrame to a numpy array
font_embeddings_array = df.values


#%%
font_embeddings_path = './data/embeddings/all_font_embeddings.npz'
font_embeddings_array = load_npz(file_path= font_embeddings_path)

dictionary_path = './data/embeddings/font_name_to_index.pickle'
dict_font_labels_to_indices= load_data_dict(dictionary_path)


#dict_font_labels_to_indices = {i: i for i in range(font_embeddings_array.shape[0])}
dict_font_indices_to_labels = {v: k for k, v in dict_font_labels_to_indices.items()}

metrics = ['angular', 'euclidean', 'manhattan', 'hamming', 'dot']

font_vector_db = MultiMetricDatabase(dimensions=font_embeddings_array.shape[1], metrics= metrics, n_trees=30)

# Add all fonts to vector database
font_vector_db.add_vectors(font_embeddings_array, dict_font_labels_to_indices)

# %%

font_candidates = get_similar_fonts(chosen_font_label=0, distance_metric='euclidean')
list_of_font_candidates = [
    {"value": label, "name": label}
    for label in font_candidates
]

print(list_of_font_candidates)
# %%
