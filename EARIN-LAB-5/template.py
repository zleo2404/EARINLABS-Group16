import os
from abc import abstractmethod, ABC
from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative) -> np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, size=(input_size, output_size))
        self.bias = np.zeros((1,output_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x 
        output = x @ self.weights + self.bias
        
        return output

    def backward(self, output_error_derivative) -> np.ndarray:
        
        input_error_derivative = output_error_derivative @ self.weights.T
        
        weights_gradient = self.x.T @ output_error_derivative
        
        bias_gradient = np.sum(output_error_derivative, axis=0, keepdims=True)
        
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * bias_gradient
        
        return input_error_derivative


class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        
        clipped_x = np.clip(x, -500.0, 500.0)
        self.output = 1.0 / (1.0 + np.exp(-clipped_x))
        
        return self.output

    def backward(self, output_error_derivative) -> np.ndarray:
        
        sigmoid_derivative = self.output * (1.0 - self.output)
        
        return output_error_derivative * sigmoid_derivative


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, output_error_derivative) -> np.ndarray:
        
        tanh_derivative = 1.0 - (self.output ** 2)
        
        return output_error_derivative * tanh_derivative

 
class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, output_error_derivative) -> np.ndarray:
        relu_derivative = (self.x > 0).astype(float)
        return output_error_derivative * relu_derivative

   
class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, output_error_derivative) -> np.ndarray:
        
        leaky_derivative = np.where(self.x > 0, 1.0, self.alpha)
        
        return output_error_derivative * leaky_derivative


class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_function(y_pred, y_true)

    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.loss_function_derivative(y_pred, y_true)

class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss_obj = None
        
        for layer in self.layers:
            layer.learning_rate = self.learning_rate
            

    def compile(self, loss: Loss) -> None:
        self.loss_obj = loss

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        # Il forward pass della rete è come una corsa a staffetta.
        # L'output del primo layer diventa l'input del secondo, e così via.
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            learning_rate: float,
            verbose: int = 0) -> List[float]:
        
        # Aggiorniamo il learning rate di tutti i layer (nel caso sia cambiato)
        for layer in self.layers:
            layer.learning_rate = learning_rate

        # Questa lista ci servirà per il tuo report: salverà l'errore ad ogni epoca 
        # così potrai plottare il grafico finale delle loss curve!
        history = [] 

        for epoch in range(epochs):
            # 1. FORWARD PASS: calcoliamo le predizioni della rete
            y_pred = self.__call__(x_train)

            # 2. CALCOLO DELLA LOSS: quanto abbiamo sbagliato?
            error = self.loss_obj.loss(y_pred, y_train)
            history.append(error)

            # 3. BACKWARD PASS: propagazione dell'errore
            # Calcoliamo la derivata iniziale (dE/dY) usando le predizioni e i valori veri
            output_error_derivative = self.loss_obj.loss_derivative(y_pred, y_train)
            
            # Viaggiamo all'indietro! Usiamo reversed() per partire dall'ultimo layer
            for layer in reversed(self.layers):
                output_error_derivative = layer.backward(output_error_derivative)

            # 4. LOGGING: stampiamo i risultati se richiesto
            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {error:.6f}")

        return history
    

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2.0 * (y_pred - y_true) / y_pred.shape[0]


def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calcola l'accuratezza confrontando l'indice del valore massimo (la classe predetta)"""
    # np.argmax trova l'indice del neurone con il valore più alto (la cifra predetta)
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

def run_experiment(activation_class: callable, 
                   X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray, 
                   hidden_neurons: int = 128, 
                   epochs: int = 15, 
                   n_seeds: int = 3) -> tuple:
    """Esegue il training su più seed e restituisce metriche e curve di loss medie."""
    
    accuracies = []
    all_losses = []

    for seed in range(n_seeds):
        # 1. IMPOSTIAMO IL SEED (Requisito fondamentale!)
        np.random.seed(seed)
        
        # 2. CREIAMO LA RETE (Architettura fissa per questo test)
        layers = [
            FullyConnected(784, hidden_neurons),
            activation_class(),           # Inseriamo la funzione di attivazione dinamica
            FullyConnected(hidden_neurons, 10),
            Sigmoid()                     # Output layer (MSE lavora bene con probabilità 0-1)
        ]
        
        net = Network(layers, learning_rate=0.1)
        net.compile(Loss(mse, mse_derivative))
        
        # 3. ADDESTRAMENTO
        # verbose=0 perché non vogliamo intasare la console con 3 seed x 4 funzioni
        history = net.fit(X_train, y_train, epochs=epochs, learning_rate=0.1, verbose=0)
        all_losses.append(history)
        
        # 4. VALUTAZIONE SUL TEST SET
        y_test_pred = net(X_test)
        acc = calculate_accuracy(y_test_pred, y_test)
        accuracies.append(acc)
        
    # Calcoliamo Media (mean) e Deviazione Standard (std) richieste
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Facciamo la media delle curve di loss lungo l'asse dei seed
    avg_loss_curve = np.mean(all_losses, axis=0)
    
    return mean_acc, std_acc, avg_loss_curve


# ---------------------------------------------------------------------------
# Data loading skeleton (shared by all groups)
# ---------------------------------------------------------------------------
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
#
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)   # Group A / C / D / E
# # OR
# fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)  # Group B
#
# X = dataset.data / 255.0
# y = dataset.target.astype(int)
# y_onehot = np.eye(10)[y]
# X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=0)


if __name__ == "__main__":
    # --- 1. CARICAMENTO DATI ---
    print("Caricamento di MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0  
    y = mnist.target.astype(int)
    y_onehot = np.eye(10)[y] 
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=0)

    # --- 2. SETUP DELL'ESPERIMENTO ---
    activations = [Sigmoid, Tanh, ReLU, LeakyReLU]
    
    loss_curves = {}
    # NUOVO: Creiamo un dizionario per salvare le accuratezze di ogni classe
    activation_accuracies = {} 

    print("\n--- INIZIO ESPERIMENTI (3 Seed per ogni attivazione) ---")
    
    for act_class in activations:
        act_name = act_class.__name__
        print(f"Addestramento con {act_name} in corso...")
        
        # Ricordati di rimuovere [:5000] quando farai il run finale!
        mean_acc, std_acc, avg_loss = run_experiment(
            activation_class=act_class,
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test,
            epochs=20
        )
        
        loss_curves[act_name] = avg_loss
        # Salviamo la CLASSE come chiave e l'accuratezza come valore
        activation_accuracies[act_class] = mean_acc 
        
        print(f"Risultato {act_name}: Test Accuracy = {mean_acc:.4f} ± {std_acc:.4f}\n")

    # --- TROVIAMO LA MIGLIORE DINAMICAMENTE ---
    # Questa fantastica funzione Python cerca nel dizionario e restituisce la CHIAVE
    # (cioè la classe dell'attivazione) che ha il VALORE (accuratezza) più alto.
    best_act_class = max(activation_accuracies, key=activation_accuracies.get)
    best_act_name = best_act_class.__name__
    
    print(f"🌟 La funzione di attivazione vincitrice è: {best_act_name}! 🌟")

    # --- 3. SECONDA ARCHITETTURA (Confronto Profondità con la Migliore) ---
    print(f"\n--- TEST SECONDA ARCHITETTURA ({best_act_name}) ---")
    print("Testiamo una rete più profonda (2 Layer Nascosti)...")
    
    accuracies_deep = []
    
    for seed in range(3):
        np.random.seed(seed)
        
        # Usiamo 'best_act_class()' per istanziare dinamicamente la vincitrice!
        deep_layers = [
            FullyConnected(784, 128),
            best_act_class(),           # <--- Inserita dinamicamente
            FullyConnected(128, 64),    # Nuovo layer
            best_act_class(),           # <--- Inserita dinamicamente
            FullyConnected(64, 10),
            Sigmoid()
        ]
        
        net_deep = Network(deep_layers, learning_rate=0.1)
        net_deep.compile(Loss(mse, mse_derivative))
        
        print(f"Addestramento architettura profonda (Seed {seed+1}/3)...")
        net_deep.fit(X_train, y_train, epochs=20, learning_rate=0.1, verbose=0)
        
        y_test_pred_deep = net_deep(X_test)
        acc_deep = calculate_accuracy(y_test_pred_deep, y_test)
        accuracies_deep.append(acc_deep)
        
    mean_acc_deep = np.mean(accuracies_deep)
    std_acc_deep = np.std(accuracies_deep)
    
    print(f"\nRisultato Architettura Profonda ({best_act_name}): Test Accuracy = {mean_acc_deep:.4f} ± {std_acc_deep:.4f}")

    # --- 4. PLOT DELLE LOSS CURVES (Alla fine di tutto) ---
    # --- 4. PLOT DELLE LOSS CURVES ---
    plt.figure(figsize=(10, 6))
    
    for act_name, loss_curve in loss_curves.items():
        # Se è LeakyReLU, la facciamo tratteggiata così si vede la ReLU sotto!
        if act_name == 'LeakyReLU':
            plt.plot(range(1, len(loss_curve) + 1), loss_curve, label=act_name, 
                     marker='x', markersize=5, linestyle='--')
        # Se è ReLU, la facciamo un po' più spessa
        elif act_name == 'ReLU':
            plt.plot(range(1, len(loss_curve) + 1), loss_curve, label=act_name, 
                     marker='o', markersize=6, linewidth=3)
        else:
            plt.plot(range(1, len(loss_curve) + 1), loss_curve, label=act_name, 
                     marker='o', markersize=4)

    plt.xlabel('Epoche')
    plt.ylabel('MSE Loss Media')
    plt.title('Confronto Funzioni di Attivazione su MNIST (Media su 3 seed)')
    plt.legend()
    plt.grid(True)
    plt.show()