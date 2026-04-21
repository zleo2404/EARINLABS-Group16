import numpy as np
import matplotlib.pyplot as plt


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
    def __init__(self, length, array=None):  # if array is None, initialize with a random binary vector
        pass

    def decode(self, lower_bound, upper_bound, aoi):
        pass

    def mutation(self, probability):
        pass

    def crossover(self, other):
        pass


# TODO: implement your group's objective function here
def objective_function(*args):
    pass


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

    def eval_objective_func(self, chromosome):
        pass

    def tournament_selection(self):
        pass

    def reproduce(self, parents):
        pass

    def plot_func(self, trace):
        pass

    def run(self):
        pass


# TODO: fill in the parameters for your group and uncomment to run
# ga = GeneticAlgorithm(
#     chromosome_length=...,
#     obj_func_num_args=2,
#     objective_function=objective_function,
#     aoi=[...],
#     population_size=...,
#     tournament_size=2,
#     mutation_probability=0.05,
#     crossover_probability=0.8,
#     num_steps=...
# )
# ga.run()