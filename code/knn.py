from sklearn.neighbors import KNeighborsClassifier
from read_data import read_data, read_test_data
from accuracy_metrics import cls_report
import time


# Perform k-NN classification on the CIFAR-10 dataset before and after PCA
def knn_experiment(training_data, training_labels, test_data, test_labels, neighbors):
    # print(data.shape)                             
    # Create and fit k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(training_data, training_labels)
    # Predict labels for test data
    predicted_labels = knn.predict(test_data)
    # Calculate accuracy
    class_report = cls_report(test_labels, predicted_labels)

    return class_report


def run_knn(training_data, training_labels, test_data, test_labels, neighbors):
    # Perform k-NN classification on the CIFAR-10 dataset before and after PCA
    print(f'Number of neighbors = {neighbors}')
    print('----------------------')

    class_report = knn_experiment(training_data, training_labels, test_data, test_labels, neighbors)
    
    print(class_report)
    


def main():
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()

    # Run k-NN classification
    # Time the k-NN classification
    start = time.time()
    run_knn(training_data, training_labels, test_data, test_labels, 1)
    end = time.time()
    print(f'k-NN classification took {end - start} seconds')
    start = time.time()
    run_knn(training_data, training_labels, test_data, test_labels, 3)
    end = time.time()
    print(f'k-NN classification took {end - start} seconds')
    



if __name__ == '__main__':
    main()

