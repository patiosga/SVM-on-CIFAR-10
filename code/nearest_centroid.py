from sklearn.neighbors import NearestCentroid
from read_data import read_data, read_test_data
from accuracy_metrics import cls_report
import time


def nearest_centroid_experiment(training_data, training_labels, test_data, test_labels):
    # print(data.shape)                      
    # Create and fit k-NN classifier
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(training_data, training_labels)
    # Predict labels for test data
    predicted_labels = nearest_centroid.predict(test_data)
    # Calculate accuracy
    class_report = cls_report(test_labels, predicted_labels)
    accuracy = nearest_centroid.score(test_data, test_labels)
    return class_report, accuracy



def main():
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()
    start = time.time()
    class_report, _ = nearest_centroid_experiment(training_data, training_labels, test_data, test_labels)
    
    print(class_report)
    print("Time taken: ", time.time()-start)
    
    
 
    

if __name__ == '__main__':
    main()


 