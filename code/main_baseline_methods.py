from read_data import read_data, read_test_data
from nearest_centroid import nearest_centroid_experiment
from knn import knn_experiment
from scaler import MyScaler
from accuracy_metrics import cls_report, accuracy
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    X_train, y_train = read_data()
    X_test, y_test = read_test_data()

    # Scale the data
    scaler : MyScaler = MyScaler.load_scaler('scaler.pkl')
    X_train = scaler.normalize_data(X_train)
    X_test = scaler.normalize_data(X_test)

    # PCA data
    X_train_pca = scaler.pca_decomposition(X_train)
    X_test_pca = scaler.pca_decomposition(X_test)

    # UMAP data
    X_train_umap = scaler.umap_decomposition(X_train)
    X_test_umap = scaler.umap_decomposition(X_test)

    # Run KNN and Nearest Centroid on all the data and plot a histogram
    knn_accuracy = {'og': [], 'pca': [], 'umap': []}
    nearest_centroid_accuracy = {'og': [], 'pca': [], 'umap': []}

    # Original Data
    start_time = time.time()
    knn_accuracy['og'].append(knn_experiment(X_train, y_train, X_test, y_test, 1)[1])
    print(f"KNN(n=1) on original data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    knn_accuracy['og'].append(knn_experiment(X_train, y_train, X_test, y_test, 3)[1])
    print(f"KNN(n=3) on original data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    nearest_centroid_accuracy['og'].append(nearest_centroid_experiment(X_train, y_train, X_test, y_test)[1])
    print(f"Nearest Centroid on original data took {time.time() - start_time:.2f} seconds")

    # PCA data
    start_time = time.time()
    knn_accuracy['pca'].append(knn_experiment(X_train_pca, y_train, X_test_pca, y_test, 1)[1])
    print(f"KNN(n=1) on PCA data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    knn_accuracy['pca'].append(knn_experiment(X_train_pca, y_train, X_test_pca, y_test, 3)[1])
    print(f"KNN(n=3) on PCA data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    nearest_centroid_accuracy['pca'].append(nearest_centroid_experiment(X_train_pca, y_train, X_test_pca, y_test)[1])
    print(f"Nearest Centroid on PCA data took {time.time() - start_time:.2f} seconds")

    # UMAP data
    start_time = time.time()
    knn_accuracy['umap'].append(knn_experiment(X_train_umap, y_train, X_test_umap, y_test, 1)[1])
    print(f"KNN(n=1) on UMAP data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    knn_accuracy['umap'].append(knn_experiment(X_train_umap, y_train, X_test_umap, y_test, 3)[1])
    print(f"KNN(n=3) on UMAP data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    nearest_centroid_accuracy['umap'].append(nearest_centroid_experiment(X_train_umap, y_train, X_test_umap, y_test)[1])
    print(f"Nearest Centroid on UMAP data took {time.time() - start_time:.2f} seconds")


    labels = ['Original', 'PCA', 'UMAP']
    knn_1_scores_list = [knn_accuracy['og'][0], knn_accuracy['pca'][0], knn_accuracy['umap'][0]]
    knn_3_scores_list = [knn_accuracy['og'][1], knn_accuracy['pca'][1], knn_accuracy['umap'][1]]
    nearest_centroid_scores_list = [nearest_centroid_accuracy['og'][0], nearest_centroid_accuracy['pca'][0], nearest_centroid_accuracy['umap'][0]]

    # Plot the histogram
    x = range(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar([p - 0.2 for p in x], knn_1_scores_list, width=0.2, label='KNN(n=1)', align='center')
    plt.bar(x, knn_3_scores_list, width=0.2, label='KNN(n=3)', align='center')
    plt.bar([p + 0.2 for p in x], nearest_centroid_scores_list, width=0.2, label='Nearest Centroid', align='center')

    plt.xlabel('Data Type')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, labels)
    plt.ylim(0.72, 1.0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


