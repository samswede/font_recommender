
#%%
import numpy as np
from annoy import AnnoyIndex
import pandas as pd
import os
import pickle


""" TO DO:
            - parameter 'dimensions' is not relevant for the font embeddings, unlike the diffusion profiles
                - remove it

            

"""

def load_data_dict(file_name):
    with open(f'{file_name}.pickle', 'rb') as handle:
        loaded_dict = pickle.load(handle)
    return loaded_dict

def save_data_dict(file_name, map_labels_to_indices):
    # assuming map_labels_to_indices is your dictionary
    with open('{file_name}.pickle', 'wb') as handle:
        pickle.dump(map_labels_to_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass


class MultiMetricDatabase:
    def __init__(self, dimensions, metrics=['angular'], n_trees=10):
        self.dimensions = dimensions
        self.n_trees = n_trees
        self.metrics = metrics
        self.databases = {}

        for metric in metrics:
            index = AnnoyIndex(dimensions, metric)
            self.databases[metric] = index

    def add_vectors(self, vectors, map_labels_to_indices):
        assert vectors.shape[1] == self.dimensions, "Vectors dimension mismatch."
        
        # Create a dictionary to map labels to vectors
        self.map_labels_to_index = {}
        for label, index in map_labels_to_indices.items():
            if index < len(vectors):
                self.map_labels_to_index[label] = vectors[index]

        for metric, index in self.databases.items():
            for label, vector in self.map_labels_to_index.items():
                index.add_item(map_labels_to_indices[label], vector.tolist())
            index.build(self.n_trees)

    def nearest_neighbors(self, query, metric, k=10):
        assert metric in self.metrics, f"Metric '{metric}' is not supported."
        index = self.databases[metric]
        return index.get_nns_by_vector(query, k)


def test_multimetricdatabase():
    # Initialize test parameters
    dimensions = 10
    n_trees = 10
    metrics = ['angular', 'euclidean', 'manhattan']
    n_vectors = 1000
    n_labels = 1200  # Creating more labels than vectors

    # Generate random vectors and labels
    vectors = np.random.rand(n_vectors, dimensions).astype('float32')
    labels = [f'drug_{i}' for i in range(n_labels)]
    map_labels_to_indices = {label: i for i, label in enumerate(labels)}

    # Initialize the MultiMetricDatabase
    db = MultiMetricDatabase(dimensions=dimensions, metrics=metrics, n_trees=n_trees)

    # Add vectors to the database
    db.add_vectors(vectors, map_labels_to_indices)

    # Check that all labels that have corresponding vectors are in the map_labels_to_index dictionary
    assert set(labels[:n_vectors]) == set(db.map_labels_to_index.keys()), "Not all labels were mapped correctly."

    # Check that no extra labels are in the map_labels_to_index dictionary
    assert len(db.map_labels_to_index) == len(vectors), "Extra labels were mapped."

    # Generate a random query vector
    query = np.random.rand(dimensions).astype('float32')

    # Query the database with different metrics and check the results
    for metric in metrics:
        result = db.nearest_neighbors(query, metric=metric, k=10)
        assert len(result) == 10, f"Query with metric '{metric}' did not return correct number of results."
        assert all(isinstance(i, int) for i in result), f"Query with metric '{metric}' returned non-integer results."

    print("All tests passed.")


