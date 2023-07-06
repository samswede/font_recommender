from sklearn.decomposition import PCA
import numpy as np

def reduce_with_pca(data, n_components):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

def reconstruct_from_pca(reduced_data, pca):
    reconstructed_data = pca.inverse_transform(reduced_data)
    return reconstructed_data
