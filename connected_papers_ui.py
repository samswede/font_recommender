
#%%
from utils import *
from dimensionality_reduction import *

import json



#%% Load data
font_embeddings_path = './data/embeddings/all_font_embeddings.npz'
font_embeddings_array = load_npz(file_path= font_embeddings_path)

dictionary_path = './data/embeddings/font_name_to_index.pickle'
dict_font_labels_to_indices= load_data_dict(dictionary_path)

#%%  Dimensionality reduction

n_components = 2

tsne_reduced_data, tsne = reduce_with_tsne(data= font_embeddings_array[0:50, :], n_components= n_components)
#pca_reduced_data, pca = reduce_with_pca(data= font_embeddings_array, n_components= n_components)

#%%
n_components = 2
plot_data_with_kmeans(data= font_embeddings_array, n_components= n_components, method='tsne', random_state=42)

