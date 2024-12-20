# Find example images where the main model is wrong and right

# import main_model as mm
import matplotlib.pyplot as plt
import numpy as np
from read_data import read_test_data, read_data
from scaler import MyScaler
import variables as var
from scaler import MyScaler
from sklearn.svm import SVC

def plot_examples(og_test_data, test_labels, y_pred):
    # Get the class labels
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
    # Plot the first 3 wrong predictions
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for i in range(3):
        axs[0, i].imshow(og_test_data[wrong_indices[i]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[0, i].set_title(f'Wrong - Pred: {class_name[y_pred[wrong_indices[i]]]}, True: {class_name[test_labels[wrong_indices[i]]]}', fontsize=16)
        axs[0, i].axis('off')

    for i in range(3):
        axs[1, i].imshow(og_test_data[right_indices[i]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[1, i].set_title(f'Right - Pred: {class_name[y_pred[right_indices[i]]]}, True: {class_name[test_labels[right_indices[i]]]}', fontsize=16)
        axs[1, i].axis('off')

    plt.show()

def find_classification_examples():
    # Load the test data
    train_data, train_labels = read_data(mask=True)
    test_data, test_labels = read_test_data(mask=True)
    og_test_data = test_data.copy()
    # Create scalers
    scaler: MyScaler = MyScaler.load_scaler('scaler.pkl')

    # Preprocess the data
    train_data = scaler.normalize_data(train_data)
    test_data = scaler.normalize_data(test_data)
    train_data_pca = scaler.pca_decomposition(train_data)
    test_data_pca = scaler.pca_decomposition(test_data)
    train_data_umap = scaler.umap_decomposition(train_data)
    test_data_umap = scaler.umap_decomposition(test_data)

    # SVM train, predict and plot
    clf_poly_umap = SVC(C=1, kernel='poly', degree=5)
    clf_poly_umap.fit(train_data_umap, train_labels)
    y_pred_poly = clf_poly_umap.predict(test_data_umap)
    print(f'UMAP SVM accuracy: {np.mean(y_pred_poly == test_labels)}')

    plot_examples(og_test_data, test_labels, y_pred_poly)


    
    clf_rbf_pca = SVC(C=10, kernel='rbf', gamma=0.001)
    clf_rbf_pca.fit(train_data_pca, train_labels)
    y_pred_rbf = clf_rbf_pca.predict(test_data_pca)
    print(f'PCA SVM accuracy: {np.mean(y_pred_rbf == test_labels)}')

    plot_examples(og_test_data, test_labels, y_pred_rbf)


def main():
    find_classification_examples()


if __name__ == '__main__':
    main()