import numpy as np
import matplotlib.pyplot as plt


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
    def __init__(self, length, array=None):  # if array is None, initialize with a random binary vector
        self.length = length
        
        if array is None:
            self.array = np.random.randint(0,2, size=length)
        else:
            self.array = np.array(array)

    def decode(self, lower_bound, upper_bound, aoi):
        segment = self.array[lower_bound:upper_bound]
        
        bit_string = "".join(segment.astype(str))
        decimal_value = int(bit_string, 2)
        
        k = upper_bound - lower_bound
        max_val = (2**k) - 1 
        
        return min_max_norm(decimal_value, 0, max_val, aoi[0], aoi[1])
            
        

    def mutation(self, probability):
        if np.random.rand() < probability:
            idx = np.random.randint(self.length)
            self.array[idx] = 1 - self.array[idx]

    def crossover(self, other):
        cutidx = np.random.randint(0,self.length)
        
        child1_array = np.concatenate((self.array[:cutidx], other.array[cutidx:]))
        child2_array = np.concatenate((other.array[:cutidx],self.array[cutidx:]))
        
        return Chromosome(self.length, child1_array), Chromosome(self.length, child2_array)


def objective_function(*args):
    x1 = args[0]
    x2 = args[1]
    
    term1 = 1.5
    term2 = np.exp(-x1**2 - x2**2)
    term3 = 0.5 * np.exp(-(x1 - 1)**2 - (x2+2)**2)
    
    return term1 - term2 - term3


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi,
                 population_size=100, tournament_size=2, mutation_probability=0.05,
                 crossover_probability=0.8, num_steps=50):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_length = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps
        self.population = []
        self.fitnesses = []

    def eval_objective_func(self, chromosome):
        args = []
        
        for i in range(self.obj_func_num_args):
            
            lower_bound = i * self.bits_per_arg
            upper_bound = (i+1) * self.bits_per_arg
            
            decoded_val = chromosome.decode(lower_bound,upper_bound,self.aoi)
            
            args.append(decoded_val)
        
        return self.objective_function(*args)
            

    def tournament_selection(self):
        parents = []
        
        for _ in range(self.population_size):
            #selfpopulation is the max value, tournament the numbers that i have to exctract and replace its without reinseriment
            candidates_idx = np.random.choice(self.population_size, self.tournament_size, replace=False)
            
            best_idx = candidates_idx[np.argmin([self.fitnesses[i] for i in candidates_idx])]
            
            parents.append(self.population[best_idx])
            
        return parents

    def reproduce(self, parents):
        next_generation = []
        
        for i in range(0,len(parents),2):
            
            p1 = parents[i]
            p2 = parents[i+1] if (i+1) < len(parents) else parents[0]
            
            if( np.random.rand() < self.crossover_probability):
                c1,c2 = p1.crossover(p2)
            else:
                c1 = Chromosome(self.chromosome_length, p1.array.copy())
                c2 = Chromosome(self.chromosome_length, p2.array.copy())
                
            c1.mutation(self.mutation_probability)
            c2.mutation(self.mutation_probability)
            
            next_generation.extend([c1,c2])
        
        return next_generation[:self.population_size]

    def plot_func(self, trace):
        
        x = np.linspace(self.aoi[0], self.aoi[1], 100)
        y = np.linspace(self.aoi[0], self.aoi[1], 100)
        #coordinates
        X, Y = np.meshgrid(x, y)
        
        #calculate fitness for each point
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.objective_function(X[i, j], Y[i, j])

        #topographic map, virdis use colours from violet to yellow
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, label='Objective Value')

        #we extract only x1 and x2
        points = np.array([t[0] for t in trace])

        #1 is dark red and 0.2 is light red, the path of the algorithm
        colors = np.linspace(1, 0.2, len(points)) # 1 è scuro, 0.2 è chiaro
        
        plt.scatter(points[:, 0], points[:, 1], c=colors, cmap='Reds', edgecolors='black', s=50, zorder=5)

        plt.title('Objective Function Contour and Best Individual Trace')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def run(self):
        self.population = [Chromosome(self.chromosome_length) for _ in range(self.population_size)]

        trace = []
        
        #how many new generations i want
        for step in range(self.num_steps):
            self.fitnesses = [self.eval_objective_func(c) for c in self.population]
            
            best_idx = np.argmin(self.fitnesses)
            best_chrom = self.population[best_idx]
            
            best_args = []
            for i in range(self.obj_func_num_args):
                best_args.append(best_chrom.decode(i * self.bits_per_arg, (i+1) * self.bits_per_arg, self.aoi))
                
            trace.append((best_args, self.fitnesses[best_idx]))
            parents = self.tournament_selection()
                
            self.population = self.reproduce(parents)
        self.plot_func(trace)
        return trace


if __name__ == "__main__":
    ga = GeneticAlgorithm(
        chromosome_length=32,
        obj_func_num_args=2,
        objective_function=objective_function,
        aoi=[-5, 5],
        population_size=100,
        tournament_size=3,
        mutation_probability=0.05,
        crossover_probability=0.8,
        num_steps=50               
    )
    
    print("Avvio dell'Algoritmo Genetico in corso...")
    final_trace = ga.run()
    print("Esecuzione terminata. Chiudi il grafico per concludere il programma.")