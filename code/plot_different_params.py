import main_model as mm
import time
import matplotlib.pyplot as plt
import json

def plot_batch_sizes():
    batch_values = [2**i for i in range(2, 8)]
    test_results = []
    training_results = []
    time_results = []
    for value in batch_values:
        trainer = mm.Model_trainer(epochs=8, batch_size=value)
        start = time.time()
        test_results.append(trainer.run(load_model=False))
        end = time.time()
        time_results.append(end - start)
        training_results.append(trainer.train_acc[-1])
    print(test_results)
    print(training_results)
    print(time_results)
    # Plotting
    plt.plot(batch_values, test_results, label="Test results")
    plt.plot(batch_values, training_results, label="Training results")
    plt.xscale('log', base=2)  # αλλάζω την κλίμακα του x αξονα γιατί αλλιώς δεν φαίνονται καλά τα αποτελέσματα
    plt.title("Results for different batch sizes")
    plt.xlabel("Batch size")
    plt.ylabel("Results")
    plt.legend()
    plt.show()

    plt.plot(batch_values, time_results)
    plt.xscale('log', base=2)
    plt.title("Time results for different batch sizes")
    plt.xlabel("Batch size")
    plt.ylabel("Time results")
    plt.show()



def plot_hidden_layer_sizes():
    n_values = [2**i for i in range(1, 6)]  # [2, 4, 8, 16, 32]
    test_results = []
    training_results = []
    time_results = []
    for neurons in n_values:
        trainer = mm.Model_trainer(epochs=8, neurons=neurons)
        start = time.time()
        test_results.append(trainer.run(load_model=False))
        end = time.time()
        time_results.append(end - start)
        training_results.append(trainer.train_acc[-1])
    print(test_results)
    print(training_results)
    print(time_results)
    # Plotting
    plt.plot(n_values, test_results, label="Test results")
    plt.plot(n_values, training_results, label="Training results")
    plt.xscale('log')
    plt.xscale('log', base=2)
    plt.title("Results for different hidden layer sizes")
    plt.xlabel("Neurons")
    plt.ylabel("Results")
    plt.legend()
    plt.show()

    plt.plot(n_values, time_results)
    plt.xscale('log', base=2)
    plt.title("Time results for different hidden layer sizes")
    plt.xlabel("Neurons")
    plt.ylabel("Time results")
    plt.show()


def plot_learning_rates():
    
    lr_values = [10**i for i in range(-5, 0)]
    test_results = []
    training_results = []
    time_results = []
    for lr in lr_values:
        trainer = mm.Model_trainer(epochs=8, learning_rate=lr)
        start = time.time()
        test_results.append(trainer.run(load_model=False))
        end = time.time()
        time_results.append(end - start)
        training_results.append(trainer.train_acc[-1])
    print(test_results)
    print(training_results)
    print(time_results)
    # Plotting
    plt.plot(lr_values, test_results, label="Test results")
    plt.plot(lr_values, training_results, label="Training results")
    plt.xscale('log')  # αλλάζω την κλίμακα του x αξονα γιατί αλλιώς δεν φαίνονται καλά τα αποτελέσματα	
    plt.title("Results for different learning rates")
    plt.xlabel("Learning rate")
    plt.ylabel("Results")
    plt.legend()
    plt.show()

    plt.plot(lr_values, time_results)
    plt.xscale('log')
    plt.title("Time results for different learning rates")
    plt.xlabel("Learning rate")
    plt.ylabel("Time results")
    plt.show()


def plot_effect_of_dropout_batch_normalization():  # με comment out τον κώδικα που έχει στην κλάση CIFAR10Model
    # ΠΡΟΫΠΟΛΟΓΙΣΜΕΝΑ ΑΠΟΤΕΛΕΣΜΑΤΑ
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    # ΠΡΟΒΟΛΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
    epochs = range(10)
    # Without dropout
    training_acc = results.get('train_acc_nothing', [])
    val_acc = results.get('val_acc_nothing', [])

    # With dropout
    training_acc_dropout = results.get('train_acc_only_dropout', [])
    val_acc_dropout = results.get('val_acc_only_dropout', [])


    # Training and validation accuracy
    plt.plot(epochs, training_acc, 'o--', label='Training Accuracy w/o Dropout', color = 'lightblue')
    plt.plot(epochs, val_acc, 'o--', label='Validation Accuracy w/o Dropout', color = 'red')
    plt.plot(epochs, training_acc_dropout, label='Training Accuracy with Dropout', color = 'lightblue')
    plt.plot(epochs, val_acc_dropout, label='Validation Accuracy with Dropout', color = 'red')
    plt.title("Training set and validation set accuracy progression")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    # With batch normalization
    training_acc_batch = results.get('train_acc_only_batch', [])
    val_acc_batch = results.get('val_acc_only_batch', [])

    # Plotting
    plt.plot(epochs, training_acc, 'o--', label='Training Accuracy w/o Batch Normalization', color = 'lightblue')
    plt.plot(epochs, val_acc, 'o--', label='Validation Accuracy w/o Batch Normalization', color = 'red')
    plt.plot(epochs, training_acc_batch, label='Training Accuracy with Batch Normalization', color = 'lightblue')
    plt.plot(epochs, val_acc_batch, label='Validation Accuracy with Batch Normalization', color = 'red')
    plt.title("Training set and validation set accuracy progression")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    # With both dropout and batch normalization
    training_acc_all = results.get('train_acc_all', [])
    val_acc_all = results.get('val_acc_all', [])

    # Plotting
    plt.plot(epochs, training_acc, 'o--', label='Training Accuracy w/o Dropout and Batch Normalization', color = 'lightblue')
    plt.plot(epochs, val_acc, 'o--', label='Validation Accuracy w/o Dropout and Batch Normalization', color = 'red')
    plt.plot(epochs, training_acc_all, label='Training Accuracy with Dropout and Batch Normalization', color = 'lightblue')
    plt.plot(epochs, val_acc_all, label='Validation Accuracy with Dropout and Batch Normalization', color = 'red')
    plt.title("Training set and validation set accuracy progression")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



def main():
    # Comment out όποιο από τα παρακάτω δεν θέλετε να τρέξετε (τα πρώτα τρία εκπαιδεύουν από την αρχή μοντέλα οπότε ειναι πιο χρονοβόρα)
    # plot_batch_sizes()
    plot_hidden_layer_sizes()
    # plot_learning_rates()
    # plot_effect_of_dropout_batch_normalization()


if __name__ == '__main__':
    main()