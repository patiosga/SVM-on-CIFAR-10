from main_model import Model_trainer, CIFAR10Model
import torch.nn as nn
import torch
import numpy as np
from accuracy_metrics import cls_report

from preprocess_data import Preprocessed_data
from read_data import read_data, read_test_data
from main_model import Model_trainer, CIFAR10Model
       
    

class One_vs_all(Model_trainer):
    def __init__(self, model=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.all_models = [CIFAR10Model(output_size=1) for _ in range(10)]  # 10 μοντέλα, ένα για κάθε κλάση


    def process_data(self, chosen_class: int, first_process=True, for_testing=False):
        if first_process:
            super().process_data()
            self.original_y_train = self.y_train.clone()  # κρατάω τα αρχικά labels γιατί πρέπει να τα αλλάζω σε κάθε μοντέλο
            self.original_y_val = self.y_val.clone()
            self.original_y_test = self.y_test.clone()

        if for_testing:
            # Το αλλάζω μόνο για το test set
            self.y_test = np.where(self.original_y_test == chosen_class, 1, 0)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
            return  # δεν αλλάζω τα labels του train και val set αφού κάνω μόνο testing
        
        # βάζω 1 στην κλάση που θέλουμε να διακρίνουμε, 0 αλλιώς
        # θέλω originals!!!
        self.y_val = np.where(self.original_y_val == chosen_class, 1, 0)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)

        self.y_train = np.where(self.original_y_train == chosen_class, 1, 0)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)

        # one hot --> y_train γιατί έχω 1 output neuron
        self.one_hot_y_train = self.y_train.unsqueeze(1)  # για να ειναι 12χ1 οπως το y_pred και να μην βγαζει ερορ η loss function
        

    def train(self):
        for i in range(10):
            if i == 0:  # για να ξέρω αν θα κάνω processing κανονικό από την αρχή ή απλά θα αλλάγξω τα lables για το τρέχον iteration εκπαίδευσης
                first_process = True
            else:
                first_process = False

            print(f"Training model for class {i}")
            print("================================")
            self.process_data(chosen_class=i, first_process=first_process)
            self.model = self.all_models[i]
            super().train()


    def test_each_model(self):
        for i in range(10):
            print(f"Testing model for class {i}")
            self.model = self.all_models[i]
            self.model.eval()  # Ensure the model is in evaluation mode
            self.process_data(chosen_class=i, first_process=False, for_testing=True)

            with torch.no_grad():
                y_pred = self.model(self.X_test)
            y_pred = torch.sigmoid(y_pred)  # Probability of the positive class (class i)
            y_pred = torch.round(y_pred).squeeze()

            report = cls_report(self.y_test, y_pred)
            print(f"Classification report for model {i}:")
            print(report)

            

    def predict(self, X: torch.Tensor):
        # Store probabilities for all classes
        predictions = []

        for i, model in enumerate(self.all_models):
            # Predict for the current class
            model.eval()  # Ensure the model is in evaluation mode
            with torch.no_grad():
                y_pred = model(X)  # Raw logits or probabilities

            # Apply sigmoid if logits are used (for binary classification)
            y_pred = torch.sigmoid(y_pred)  # Probability of the positive class (class i)

            # Append to predictions list
            predictions.append(y_pred)

        # Convert to a single tensor: shape (num_samples, num_classes)
        predictions = torch.stack(predictions, dim=1)
        print(predictions.shape)
        print(predictions[:10])

        # Final predictions: select the class with the highest probability
        predictions = torch.argmax(predictions, dim=1)
        return predictions.squeeze() # Remove the extra dimension
        


def main(): 
    trainer = One_vs_all(loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0])), epochs=1)
    # Χρησιμοποιώ BCEWithLogitsLoss γιατί έχω binary classification και θέλω να επιβαρύνω το loss της μικρής κλάσης επι 9
    trainer.train()
    # trainer.process_data(chosen_class=0, first_process=True)
    trainer.test_each_model()
    predictions = trainer.predict(trainer.X_test)
    print(trainer.original_y_test[:20])
    print(predictions[:20])
    report = cls_report(trainer.original_y_test, predictions)
    print("Classification report for all models:")
    print(report)


if __name__ == "__main__":
    main()

    
        
            