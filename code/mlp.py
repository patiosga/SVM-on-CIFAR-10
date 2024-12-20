import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

from accuracy_metrics import cls_report
from scaler import MyScaler
from read_data import read_data, read_test_data


class CIFAR10Model(nn.Module):
    '''
    MLP ενός κρυφού επιπέδου
    ίδιος σκελετός κώδικας με πρώτη εργασία'''
    def __init__(self, model=None, input_size=3072, output_size=10):
        super(CIFAR10Model, self).__init__()
        if model is None:
            self.model = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),

                nn.Linear(64, output_size),
                nn.Softmax(dim=1)
            )
        else: 
            self.model = model

    def forward(self, x):
        return self.model(x)
    
    def __str__(self):
        return str(self.model)
    

class Model_trainer:
    def __init__(self, epochs=5, batch_size=16, model=None, loss_fn=None, optimizer=None, scheduler=None, input_size=3072, learning_rate=0.001):
        '''
        Κατασκευαστής για την κλάση Model_trainer
        Αρχικοποιεί τα epochs, batch_size, μοντέλο, συνάρτηση απώλειας, βελτιστοποιητή και scheduler
        '''
        self.epochs = epochs
        self.batch_size = batch_size

        # Ορίζω το μοντέλο, τη συνάρτηση απώλειας, τον βελτιστοποιητή και τον scheduler
        # Το αφήνω να δημιουργηθεί από τον χρήστη αν δεν δοθεί για λόγους debugging
        if model is None:
            self.model = CIFAR10Model(input_size=input_size)
        else:
            self.model = model
        
        if loss_fn is None:
            self.loss_fn = nn.MultiMarginLoss()
        else:
            self.loss_fn = loss_fn

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, threshold=0.005)
        else:
            self.scheduler = scheduler
        
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.LR_values = []



    def train(self, X_train, y_train, X_val, y_val, one_hot_y_train=None):
        '''
        Εκπαιδεύει το μοντέλο και κρατάει τα αποτελέσματα για το training και το validation set
        '''
        batches_per_epoch = len(X_train) // self.batch_size

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            for i in range(batches_per_epoch):
                start = i * self.batch_size  # Αρχή του batch
                Xbatch = X_train[start : start + self.batch_size]
                ybatch = y_train[start : start + self.batch_size]  
                # παίρνω τα one-hot labels για υπολογισμό του loss αλλιώς χρησιμοποιώ κανονικά labels

                # Υπολογισμός του loss και backpropagation
                y_pred = self.model(Xbatch)
                loss = self.loss_fn(y_pred, ybatch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Υπολογισμός του loss και της ακρίβειας στο validation set και το training set μετά το τέλος κάθε εποχής
            with torch.no_grad():
                y_pred_val = self.model(X_val)
                y_pred_train = self.model(X_train)

            loss_train = self.loss_fn(y_pred_train, y_train)
            self.train_loss.append(float(loss_train))

            y_pred_val = torch.argmax(y_pred_val, dim=1)
            y_pred_train = torch.argmax(y_pred_train, dim=1)

            accuracy_val = (y_pred_val == y_val).float().mean()
            accuracy_train = (y_pred_train == y_train).float().mean()

            print("LR value used:", self.scheduler.get_last_lr()[0])  # επέστρεφε λίστα
            self.LR_values.append(self.scheduler.get_last_lr()[0])
            # Μειώνω LR αν χρειάζεται
            self.scheduler.step(accuracy_val)  # μειώνω το learning rate με βάση την ακρίβεια στο validation set

            # Τα κρατάω σε λίστες για να τα πλοτάρω στο τέλος
            self.val_acc.append(float(accuracy_val))
            self.train_acc.append(float(accuracy_train))
            print(f"End of {epoch+1}, training accuracy {self.train_acc[-1]}, validation set accuracy {self.val_acc[-1]}")


    def test(self, X_test, y_test):
        '''
        Υπολογισμός της ακρίβειας στο test set και εκτύπωση του classification report
        '''
        with torch.no_grad():
            y_pred = self.model(X_test)
        y_pred = torch.argmax(y_pred, dim=1)
        test_acc = (y_pred == y_test).float().mean()
        print(cls_report(np.array(y_test), np.array(y_pred)))
        print("Test accuracy:", test_acc)
        return test_acc


    def plot_training_progress(self):
        '''
        Πλοτάρει την ακρίβεια και το loss στο training και validation set
        '''
        epochs = range(self.epochs)
    
        # Training and validation accuracy
        plt.plot(epochs, self.train_acc, label='Training Accuracy')
        plt.plot(epochs, self.val_acc, label='Validation Accuracy')
        plt.title("Training set and validation set accuracy progression")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim((0, 1))
        plt.legend()
        plt.show()



    def write_model_to_file(self):
        torch.save(self.model.state_dict(), 'cifar10_main_model.pth')


    def load_model_from_file(self):
        self.model.load_state_dict(torch.load('cifar10_main_model.pth'))
        self.model.eval()
        return self.model


    def run(self, X_train, y_train, X_val, y_val, X_test, y_test, one_hot_y_train, load_model=False):
        if load_model:
            self.load_model_from_file()  # παίρνω το αρχείο από τον δίσκο -- by default κάνει train νέο μοντέλο
        else:
            self.train(X_train, y_train, X_val, y_val, one_hot_y_train)
            # self.plot_training_progress()  # πλοτάρω την πρόοδο του μοντέλου

        return self.test(X_test, y_test)
        
    

def main():
    # load the scaler
    scaler: MyScaler = MyScaler.load_scaler('scaler.pkl')
    # load the data
    X_train, y_train = read_data()
    # normalize the data
    X_train = scaler.normalize_data(X_train)
    print('Data loaded')

    # split the data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.6, random_state=42, stratify=y_train)
    # perform PCA
    X_train_pca = scaler.pca_decomposition(X_train)
    X_val_pca = scaler.pca_decomposition(X_val)
    # perform UMAP
    X_train_umap = scaler.umap_decomposition(X_train)
    X_val_umap = scaler.umap_decomposition(X_val)
    # load the test data
    X_test, y_test = read_test_data()
    # normalize the test data
    X_test = scaler.normalize_data(X_test)
    # perform PCA on the test data
    X_test_pca = scaler.pca_decomposition(X_test)
    # perform UMAP on the test data
    X_test_umap = scaler.umap_decomposition(X_test)

    # One hot encode the train labels
    one_hot_y_train = np.eye(10)[y_train]

    # Convert the data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    one_hot_y_train = torch.tensor(one_hot_y_train, dtype=torch.float32)
    X_train_pca = torch.tensor(X_train_pca, dtype=torch.float32)
    X_train_umap = torch.tensor(X_train_umap, dtype=torch.float32)
    X_test_pca = torch.tensor(X_test_pca, dtype=torch.float32)
    X_test_umap = torch.tensor(X_test_umap, dtype=torch.float32)
    X_val_pca = torch.tensor(X_val_pca, dtype=torch.float32)
    X_val_umap = torch.tensor(X_val_umap, dtype=torch.float32)

    print('Data preprocessed')


    # create the model - original data
    start = time.time()
    trainer_og = Model_trainer(epochs=13, input_size=X_train.shape[1])
    print(trainer_og.run(X_train, y_train, X_val, y_val, X_test, y_test, one_hot_y_train))
    print("Time taken (original data): ", time.time()-start)
    print('Model trained on original data')

    # create the model - PCA data
    start = time.time()
    trainer_pca = Model_trainer(epochs=13, input_size=X_train_pca.shape[1])
    print(trainer_pca.run(X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, one_hot_y_train))
    print("Time taken (pca): ", time.time()-start)  
    print('Model trained on PCA data')

    # create the model - UMAP data
    start = time.time()
    trainer_umap = Model_trainer(epochs=13, input_size=X_train_umap.shape[1])
    print(trainer_umap.run(X_train_umap, y_train, X_val_umap, y_val, X_test_umap, y_test, one_hot_y_train))
    print("Time taken (umap): ", time.time()-start)  
    print('Model trained on UMAP data')

    # Plot training and validation accuracy for all three models
    epochs = range(13)
    plt.plot(epochs, trainer_og.train_acc, label='Original Data - Training Accuracy', marker='o')
    plt.plot(epochs, trainer_og.val_acc, label='Original Data - Validation Accuracy', marker='o')
    plt.plot(epochs, trainer_pca.train_acc, label='PCA Data - Training Accuracy', marker='x')
    plt.plot(epochs, trainer_pca.val_acc, label='PCA Data - Validation Accuracy', marker='x')
    plt.plot(epochs, trainer_umap.train_acc, label='UMAP Data - Training Accuracy', marker='s')
    plt.plot(epochs, trainer_umap.val_acc, label='UMAP Data - Validation Accuracy', marker='s')
    plt.title("Training and Validation Accuracy for Original, PCA, and UMAP Data")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    plt.legend()
    plt.show()






if __name__ == '__main__':
    main()





