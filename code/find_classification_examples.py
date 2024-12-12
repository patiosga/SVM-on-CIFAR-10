# Find example images where the main model is wrong and right

# import main_model as mm
import matplotlib.pyplot as plt
import torch
import numpy as np
from read_data import read_test_data
from preprocess_data import Preprocessed_data
from scaler import MyScaler
import variables as var
from scaler import Scaler

def find_classification_examples():
    # Load the test data
    test_data, test_labels = read_test_data()
    og_test_data = test_data.copy()

    # Create scalers
    scaler: MyScaler = MyScaler.load_scaler('scaler.pkl')

    # Preprocess the data
    test_data = scaler.normalize_data(test_data)
    test_data_pca = scaler.pca_decomposition(test_data)

    # SVM HERE !!!!!!!!!!!!!!!!!!!!!!!!!
    # temp = mm.Model_trainer()
    # model = temp.load_model_from_file()
    # # Get the predictions
    # with torch.no_grad():
    #     predictions = model(test_data)
    predictions = []
    
    # Get the class labels
    y_pred = torch.argmax(predictions, dim=1)
    y_pred = y_pred.numpy()
    print(f'Predictions: {y_pred[:10]}')
    print(f'Test labels: {test_labels[:10]}')

    # Find the indices of the wrong predictions
    wrong_indices = np.where(y_pred != test_labels)[0]
    # Get 6 random indices
    np.random.seed(42)  # Για να παιρνω καθε φορά τα 6 ίδια indices --  επίσης αμα δεν πάρω ραντομ επιστρέφει πολλά βατράχια
    wrong_indices = np.random.choice(wrong_indices, 6, replace=False)
    # Find the indices of the right predictions
    right_indices = np.where(y_pred == test_labels)[0]
    # Get 6 random indices
    right_indices = np.random.choice(right_indices, 6, replace=False)
    
    class_name = var.labels
    # Plot the first 6 wrong predictions
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for i in range(3):
        axs[0, i].imshow(og_test_data[wrong_indices[i]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[0, i].set_title(f'Pred: {class_name[y_pred[wrong_indices[i]]]}, True: {class_name[test_labels[wrong_indices[i]]]}', fontsize=16)
        axs[0, i].axis('off')
        axs[1, i].imshow(og_test_data[wrong_indices[i + 3]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[1, i].set_title(f'Pred: {class_name[y_pred[wrong_indices[i + 3]]]}, True: {class_name[test_labels[wrong_indices[i + 3]]]}', fontsize=16)
        axs[1, i].axis('off')
    plt.show()

    # Plot the first 6 right predictions
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for i in range(3):
        axs[0, i].imshow(og_test_data[right_indices[i]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[0, i].set_title(f'Pred: {class_name[y_pred[right_indices[i]]]}, True: {class_name[test_labels[right_indices[i]]]}', fontsize=16)
        axs[0, i].axis('off')
        axs[1, i].imshow(og_test_data[right_indices[i + 3]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[1, i].set_title(f'Pred: {class_name[y_pred[right_indices[i + 3]]]}, True: {class_name[test_labels[right_indices[i + 3]]]}', fontsize=16)
        axs[1, i].axis('off')
    plt.show()


def main():
    find_classification_examples()


if __name__ == '__main__':
    main()