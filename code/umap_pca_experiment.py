import umap
import numpy as np
from read_data import read_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from variables import labels

from matplotlib.colors import ListedColormap

def pca(X_train):
    # Reduce the dimensionality of the dataset to 2 dimensions using PCA
    pca = PCA(n_components=0.9)
    X_pca_train = pca.fit_transform(X_train)
    print(f'PCA: Reduced the dimensionality to {X_pca_train.shape[1]} dimensions')
    

def plot_2D(X, y, class_pairs, custom_cmap, method):
    for pair in class_pairs:
        # Filter the data to include only the current pair of classes
        mask = (y == pair[0]) | (y == pair[1])
        X_filtered = X[mask]
        y_filtered = y[mask]

        plt.subplot(2, 3, class_pairs.index(pair) + 1)
        scatter = plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered.astype(int), cmap=custom_cmap, s=5)
        plt.colorbar(scatter, ticks=np.arange(2))
        plt.title(f'{method} projection (classes {labels[pair[0]]} and {labels[pair[1]]})')
        plt.legend(handles=scatter.legend_elements()[0], labels=[labels[pair[0]], labels[pair[1]]])

    plt.show()
    

def umap_vs_pca(plot=True, plot_silhouette=True):
    # Define a custom colormap
    custom_colors = ['#1f77b4', '#ff7f0e']
    custom_cmap = ListedColormap(custom_colors)
    class_pairs = [(1, 5), (2, 6), (4, 8), (5, 9), (7, 8), (1, 9)]

    # Load the CIFAR-10 dataset
    X, y = read_data()
    # split into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)
    # Normalize the data
    if plot:  # Plot the 2D projections
        # Reduce the dimensionality of the dataset to 2 dimensions using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)
        plot_2D(X_pca, y_train, class_pairs, custom_cmap, method='PCA')

        # Reduce the dimensionality of the dataset to 2 dimensions using UMAP
        reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric='euclidean')
        X_embedded = reducer.fit_transform(X_train)
        plot_2D(X_embedded, y_train, class_pairs, custom_cmap, method='UMAP')

    if plot_silhouette:  # Plot the silhouette scores
        # Reduce the dimensionality of the dataset to 100 dimensions using PCA --> 90% of variance is achieved
        pca = PCA(n_components=100)
        X_pca = pca.fit_transform(X_train)

        # Reduce the dimensionality of the dataset to 10 dimensions using UMAP
        reducer = umap.UMAP(n_components=10, n_neighbors=10, min_dist=0.1, metric='')
        X_embedded = reducer.fit_transform(X_train)

        # Calculate the silhouette score for the UMAP projection and the PCA projection
        plt.figure(figsize=(10, 6))
        width = 0.2  # the width of the bars

        # Create an array with the positions of the bars on the x-axis
        x = np.arange(len(class_pairs))

        # Plot the bars for each method
        plt.bar(x - width, [silhouette_score(X_train[(y_train == pair[0]) | (y_train == pair[1])], y_train[(y_train == pair[0]) | (y_train == pair[1])]) for pair in class_pairs], width, label='Original(3072)')
        plt.bar(x, [silhouette_score(X_pca[(y_train == pair[0]) | (y_train == pair[1])], y_train[(y_train == pair[0]) | (y_train == pair[1])]) for pair in class_pairs], width, label='PCA(100)')
        plt.bar(x + width, [silhouette_score(X_embedded[(y_train == pair[0]) | (y_train == pair[1])], y_train[(y_train == pair[0]) | (y_train == pair[1])]) for pair in class_pairs], width, label='UMAP(10)')

        # Add labels and title
        plt.xlabel('Class pairs')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette scores for different class pairs and methods')
        plt.xticks(x, [f'{labels[pair[0]]} vs {labels[pair[1]]}' for pair in class_pairs])
        plt.legend()

        # Show the plot
        plt.show()


def umap_components_experiment():
    # Load the CIFAR-10 dataset
    X, y = read_data()
    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)

    # Define the number of components to test
    n_components = [2, 5, 10, 20, 50, 100, 200]

    # Calculate the silhouette score for each number of components
    silhouette_scores = []
    for n in n_components:
        print(f'Calculating silhouette score for {n} components...')
        reducer = umap.UMAP(n_components=n, n_neighbors=10, min_dist=0.1, metric='euclidean')
        X_embedded = reducer.fit_transform(X_train)
        silhouette_score_n = silhouette_score(X_embedded, y_train)
        silhouette_scores.append(silhouette_score_n)

    # Plot the silhouette scores
    plt.plot(n_components, silhouette_scores, marker='o')
    plt.xticks(n_components)
    plt.xlabel('Number of components')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score vs number of components')
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    umap_vs_pca(plot=True, plot_silhouette=True)
    umap_components_experiment()