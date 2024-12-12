import numpy as np
import variables as var
from scaler import normalize_data
from sklearn.model_selection import train_test_split
from read_data import read_data, read_test_data
import pickle
import torch
class Preprocessed_data():
    def __init__(self, training_data, training_labels, test_data, test_labels):
        '''
        Initializer για δεδομένα εκπαίδευσης και test
        Κανονικοποιεί δεοδομένα και δημιουργεί one-hot labels
        '''
        self.X_train = training_data
        self.y_train = training_labels
        self.X_test = test_data
        self.y_test = test_labels  # για καποιον λογο το επαιρνε ως λιστα και οχι ως numpy array

        # Normalize data
        self.X_train = normalize_data(training_data)
        self.X_test = normalize_data(test_data)

        # One-hot labels
        self.one_hot_y_train = np.eye(var.num_of_classes)[training_labels]  # διαλέγει τις training_label γραμμές από τον μοναδιαίο πίνακα
        # self.one_hot_y_test = np.eye(var.num_of_classes)[test_labels]  # δεν χρειάζεται πουθενά


    def split_data(self, test_size: float):
        '''
        Διαχωρίζει τα δεδομένα εκπαίδευσης σε train και validation
        '''
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=test_size, random_state=42, stratify=self.y_train)
        self.one_hot_y_train = np.eye(var.num_of_classes)[self.y_train]  # ανανέωση των one-hot labels αφου έγινε split το dataset
        # self.one_hot_y_val = np.eye(var.num_of_classes)[self.y_val]  # δεν χρειάζεται πουθενά
        
    def convert_to_tensor(self):
        '''
        Στο τέλος του pipeline μετατρέπω τα δεδομένα σε torch tensors
        '''
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.one_hot_y_train = torch.tensor(self.one_hot_y_train, dtype=torch.float32)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)

        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
       

    @staticmethod
    def create_3d_data(data: np.ndarray) -> np.ndarray:
        # Create a 3D NumPy array of shape (50000, 32, 32) for the red, green, or blue channel
        data_3d = data.reshape(-1, 3, 32, 32)
        # print(data_3d.shape)  # (50000, 3, 32, 32)
        return data_3d
    

    @staticmethod
    def write_to_pickle_file(data_object: 'Preprocessed_data'):
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(data_object, f)


    @staticmethod
    def read_from_pickle_file() -> 'Preprocessed_data':
        with open('preprocessed_data.pkl', 'rb') as f:
            return pickle.load(f)

def main():
    # Write object to pickle for future use - φουυυυυλ γρηγορο
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()

    processed_data = Preprocessed_data(training_data, training_labels, test_data, test_labels)
    processed_data.split_data(test_size=0.8, one_hot=True)
    Preprocessed_data.write_to_pickle_file(processed_data)

   
if __name__ == '__main__':
    main()




