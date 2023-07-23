
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

#%%


def convert_numpy_to_visjs_format(reduced_data, labels_to_indices_dict, images_to_paths_dict):
    """
    This function converts the reduced dimensional data and corresponding labels into a format suitable for Vis.js.

    Parameters:
    reduced_data (numpy.ndarray): The reduced dimensional data obtained from t-SNE or another dimensionality reduction method.
    Each row is a sample and each column is a dimension (2D or 3D).

    labels_dict (dict): A dictionary mapping the label names to their corresponding indices in the reduced_data array.

    images_dict (dict): A dictionary mapping the label names to their corresponding image paths.
    Each key is a label and each value is a string representing the path to the image file corresponding to that label.

    Returns:
    nodes (list): A list of dictionaries where each dictionary represents a node for Vis.js.
    Each node dictionary contains the following key-value pairs:
    - 'id': the index of the sample
    - 'label': the label of the sample
    - 'shape': the shape of the node, here set as 'circularImage' for image representation
    - 'image': the path to the image file for the node
    - 'x' and 'y': the x and y coordinates of the node in the 2D plot
    - 'fixed': a dictionary specifying whether the x and y coordinates of the node are fixed. Here, both are set to True.
    """
    nodes = []
    for i, coordinates in enumerate(reduced_data):
        nodes.append({
            "id": i, 
            "label": list(labels_to_indices_dict.keys())[list(labels_to_indices_dict.values()).index(i)],  # Fetch the label corresponding to the index
            "shape": "circularImage",  # Specify the shape of the node as a circular image
            "image": images_to_paths_dict[list(labels_to_indices_dict.keys())[list(labels_to_indices_dict.values()).index(i)]],  # Fetch the image path for the label
            "x": coordinates[0],  # Specify the x-coordinate of the node
            "y": coordinates[1],  # Specify the y-coordinate of the node
            "fixed": {"x": True, "y": True},  # Set the x and y coordinates as fixed
        })
    return nodes



#%%
@app.post("/graph", response_class=JSONResponse)
async def get_graph_data(request: GraphRequest):
    # Extract parameters from request
    font_label = request.disease_label
    total_num_nodes = request.total_num_nodes

    print(f'font_label: {font_label}')
    print(f'total_num_nodes: {total_num_nodes}')

    # Generate similar fonts

    
    # Convert graph data into a format that vis.js can handle
    visjs_nodes = convert_numpy_to_visjs_format(reduced_data, labels_to_indices_dict, images_to_paths_dict)

    # Create the response
    response = {
        "MOA_network": visjs_nodes,
    }

    return response