import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningCartPole:
    def __init__(self, bins=(20, 20, 20, 20), episodes=2000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Inizializza l'algoritmo Q-Learning per il CartPole.
        """
        # Iperparametri
        self.episodes = episodes
        self.alpha = alpha                # Learning rate
        self.gamma = gamma                # Discount factor
        self.epsilon = epsilon            # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Inizializzazione ambiente
        self.env = gym.make('CartPole-v1')
        
        # Definizione dei limiti (bounds) per la quantizzazione.
        # I limiti di posizione e angolo presi dalla documentazione,
        # le velocità sono stimate poiché tecnicamente vanno da -Inf a +Inf.
        self.bins = bins
        self.bounds = [
            [-2.4, 2.4],          # Posizione carrello
            [-3.0, 3.0],          # Velocità carrello (stimata)
            [-0.2095, 0.2095],    # Angolo palo (in radianti)
            [-10.0, 10.0]         # Velocità angolare palo (stimata)
        ]
        
        # Inizializzazione Q-Table con zeri
        # Dimensioni: (bins_pos, bins_vel, bins_ang, bins_ang_vel, numero_azioni)
        q_table_shape = self.bins + (self.env.action_space.n,)
        self.q_table = np.zeros(q_table_shape)
        
        # Array per memorizzare le metriche
        self.rewards_history = []

    def discretize_state(self, state):
        """
        Converte uno stato continuo (float) in uno stato discreto (indici della Q-Table).
        """
        discrete_state = []
        for i in range(len(state)):
            # Mappa il valore continuo nel range [0, 1]
            scaling = (state[i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0])
            # Calcola l'indice del bin
            new_state = int(np.round(scaling * (self.bins[i] - 1)))
            # Assicurati che l'indice rimanga all'interno dei bordi validi
            new_state = max(0, min(self.bins[i] - 1, new_state))
            discrete_state.append(new_state)
            
        return tuple(discrete_state)

    def choose_action(self, state):
        """
        Sceglie un'azione usando la policy Epsilon-Greedy.
        """
        if np.random.random() < self.epsilon:
            # Esplorazione: azione casuale
            return self.env.action_space.sample()
        else:
            # Sfruttamento: azione con il valore Q massimo per quello stato
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        Aggiorna il valore nella Q-Table usando l'equazione di Bellman.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        
        # Aggiornamento
        self.q_table[state][action] += self.alpha * td_error

    def train(self):
        """
        Ciclo di addestramento principale.
        """
        print("Inizio addestramento...")
        for episode in range(self.episodes):
            # Il nuovo standard Gymnasium restituisce (state, info)
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            
            total_reward = 0
            done = False
            
            while not done:
                # 1. Scegli l'azione
                action = self.choose_action(state)
                
                # 2. Esegui l'azione (Gymnasium moderno restituisce 5 valori)
                next_state_continuous, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 3. Discretizza il nuovo stato
                next_state = self.discretize_state(next_state_continuous)
                
                # 4. Aggiorna la Q-Table
                self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                
            # Salva la metrica dell'episodio
            self.rewards_history.append(total_reward)
            
            # Decadimento dell'epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Stampa il progresso ogni 200 episodi
            if (episode + 1) % 200 == 0:
                avg_reward = np.mean(self.rewards_history[-200:])
                print(f"Episodio {episode + 1}/{self.episodes} - Reward medio ultimi 200: {avg_reward:.2f} - Epsilon: {self.epsilon:.3f}")
                
        print("Addestramento completato!")
        self.env.close()

    def plot_metrics(self):
        """
        Visualizza l'andamento delle ricompense per mostrare la convergenza.
        """
        # Calcola la media mobile per pulire il grafico (finestra di 50 episodi)
        window = 50
        moving_avg = np.convolve(self.rewards_history, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history, alpha=0.3, color='blue', label='Reward per Episodio')
        plt.plot(np.arange(window-1, len(self.rewards_history)), moving_avg, color='red', label=f'Media Mobile ({window} ep)')
        plt.title('Miglioramento dell\'Agente Q-Learning sul CartPole')
        plt.xlabel('Episodi')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Inizializza ed esegui l'algoritmo
    agent = QLearningCartPole(
        episodes=3000, 
        alpha=0.1, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_decay=0.998 # Decadimento leggermente più lento per esplorare meglio
    )
    agent.train()
    agent.plot_metrics()