from sklearn.preprocessing import StandardScaler
from read_data import read_data
from sklearn.decomposition import PCA
import umap
import pickle

class MyScaler():
    def __init__(self, pca_comp=0.95, data=None):
        # z-score normalization
        self.normalizer = StandardScaler(with_mean=True, with_std=True)  
        # PCA
        self.pca_comp = pca_comp
        self.pca = PCA(n_components=pca_comp)  # 95% of the variance is preserved

        # UMAP -- τα components βγήκαν από το umap_pca_experiment.py
        self.umap = umap.UMAP(n_components=50, n_neighbors=15, min_dist=0.1, metric='euclidean')

        if data is not None:
            # Fit the normalizer on the data
            self.normalizer.fit(data)

            # Normalize the data before applying dimensionality reduction - ALWAYS NORMALIZE 
            data = self.normalizer.transform(data)

            # Fit the PCA on the data
            self.pca.fit(data)
            # κρατάω τους scalers που έχει εφαρμοστεί στα training δεδομένα για να τον εφαρμόσω και στα test data
            # Fit the UMAP on the data
            self.umap.fit(data)

    def normalize_data(self, data):
        # Transform the data using the fitted scaler
        data_normalized = self.normalizer.transform(data)
        return data_normalized
    
    def pca_decomposition(self, data):
        # Perform PCA analysis
        data_pca = self.pca.transform(data)
        # προβάλλω τα οποιαδήποτε νέα δεδομένα στους άξονες που έχουν δημιουργηθεί από το PCA για το training set
        return data_pca
    
    def umap_decomposition(self, data):
        # Perform UMAP analysis
        # τα νέα σημεία που έρχονται απο το τεστ σετ θα βρουν NNs και απο τα training data και θα προβληθούν στον ίδιο χώρο
        # τα νέα δεδομένα δεν επηρεάζουν τον χώρο των training data
        data_umap = self.umap.transform(data)
        return data_umap

    def save_scaler(self, filename):
        # Save the scaler to a file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_scaler(filename):
        # Load the scaler from a file
        with open(filename, 'rb') as f:
            scaler = pickle.load(f)
        return scaler


def main():
    data, _ = read_data()
    print(data.shape)  # (50000, 3072)
    # scaler: MyScaler = MyScaler.load_scaler('scaler.pkl')
    scaler = MyScaler(data=data)
    scaler.save_scaler('scaler.pkl')

    # data_normalized = scaler.normalize_data(data)
    # print(data_normalized.shape)
    # data_pca = scaler.pca_decomposition(data_normalized)
    # print(data_pca.shape)
    # data_umap = scaler.umap_decomposition(data_normalized)
    # print(data_umap.shape)


if __name__ == '__main__':
    main()