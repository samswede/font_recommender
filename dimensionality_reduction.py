
# import necessary libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

import numpy as np



# Function for dimensionality reduction using Principal Component Analysis (PCA)
def reduce_with_pca(data, n_components):
    """
    This function reduces the dimensionality of the input data using PCA.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): The high-dimensional data to reduce. 
    This is expected to be a 2D array-like object where each row is a sample and 
    each column is a feature.
    
    n_components (int): The number of dimensions to reduce the data to. 
    This should be less than the number of features in the data.

    Returns:
    tuple: A tuple containing the reduced data as a numpy array and the fitted PCA object. 
    The reduced data can be used for further analysis or visualization, 
    and the PCA object can be used to transform additional data or to perform an inverse transform.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca


# Function for reconstructing data from its reduced form using PCA
def reconstruct_from_pca(reduced_data, pca):
    """
    This function reconstructs the original data from the reduced data using the PCA object.

    Parameters:
    reduced_data (numpy.ndarray or pandas.DataFrame): The reduced data. 
    This is expected to be a 2D array-like object where each row is a sample and 
    each column is a dimension in the reduced space.

    pca (sklearn.decomposition.PCA): The PCA object that was used to reduce the data. 
    This object contains the information necessary to transform the data back to its original space.

    Returns:
    numpy.ndarray: The reconstructed data as a numpy array. 
    This data will be in the same format as the original data passed to the PCA object.
    """
    reconstructed_data = pca.inverse_transform(reduced_data)
    return reconstructed_data


# Function for dimensionality reduction using t-Distributed Stochastic Neighbor Embedding (t-SNE)
def reduce_with_tsne(data, n_components, perplexity= 30, random_state=42):
    """
    This function reduces the dimensionality of the input data using t-SNE.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): The high-dimensional data to reduce. 
    This is expected to be a 2D array-like object where each row is a sample and 
    each column is a feature.

    n_components (int): The number of dimensions to reduce the data to. 
    This is usually set to 2 or 3 for visualization purposes.

    random_state (int, optional): The seed for the random number generator. 
    This is used to ensure that the random processes used by t-SNE produce the same result each time the function is run. 
    The default value is 42.

    Returns:
    tuple: A tuple containing the reduced data as a numpy array and the fitted t-SNE object. 
    The reduced data can be used for further analysis or visualization. 
    The t-SNE object can't be used to transform additional data or perform an inverse transform as 
    t-SNE doesn't support these operations.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    reduced_data = tsne.fit_transform(data)
    return reduced_data, tsne


def plot_2d(data, labels, title):
    """
    This function plots the input 2D data with different colors for different labels.

    Parameters:
    data (numpy.ndarray): The 2D data to plot. Each row is a sample and each column is a dimension.
    labels (numpy.ndarray): The label for each sample in the data. Used to color the points.
    title (str): The title for the plot.
    """
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.show()


def plot_data_with_kmeans(data, n_components, method='pca', random_state=42):
    """
    This function reduces the dimensionality of the data to 2D using PCA or t-SNE for visualization,
    applies a K-Means clustering to the original high dimensional data, and then plots the reduced data
    colored according to the cluster labels found.

    Parameters:
    data (numpy.ndarray): The high-dimensional data to reduce and plot.
    n_components (int): The number of dimensions to reduce the data to (should be 2 for plotting).
    method (str, optional): The dimensionality reduction method to use. Can be 'pca' or 'tsne'. Defaults to 'pca'.
    random_state (int, optional): The seed for the random number generator. Defaults to 42.
    """

    # Reduce data
    if method == 'pca':
        reduced_data, _ = reduce_with_pca(data, n_components)
    elif method == 'tsne':
        reduced_data, _ = reduce_with_tsne(data, n_components, random_state)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Finding the optimal number of clusters using the Elbow method
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=random_state)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)

    # Finding the elbow point
    kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    optimal_clusters = kn.knee

    # Label data with K-Means
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=random_state)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Plot reduced data with colors indicating the labels obtained from original data
    plot_2d(reduced_data, labels, f"Data plotted with {method.upper()} and K-Means")


