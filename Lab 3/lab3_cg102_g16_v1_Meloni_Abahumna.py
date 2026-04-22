import random

import matplotlib
import numpy as np

import matplotlib.pyplot as plt

OUTPUT_PREFIX = "lab3_cg102_g16_v1_Meloni_Abahumna"


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


def normalize_aoi(aoi, num_args):
    if len(aoi) == 2 and np.isscalar(aoi[0]) and np.isscalar(aoi[1]):
        return [tuple(map(float, aoi)) for _ in range(num_args)]
    return [tuple(map(float, bounds)) for bounds in aoi]


class Chromosome:
    def __init__(self, length, array=None):  # if array is None, initialize with a random binary vector
        self.length = length
        if array is None:
            self.array = np.random.randint(0, 2, size=length, dtype=int)
        else:
            self.array = np.array(array, dtype=int)

    def decode(self, lower_bound, upper_bound, aoi):
        segment = self.array[lower_bound:upper_bound]
        decimal_value = 0
        for bit in segment:
            decimal_value = (decimal_value << 1) | int(bit)

        return min_max_norm(decimal_value, 0, (2 ** len(segment)) - 1, aoi[0], aoi[1])

    def mutation(self, probability):
        if random.random() < probability:
            gene_idx = random.randrange(self.length)
            self.array[gene_idx] = 1 - self.array[gene_idx]

    def crossover(self, other):
        split_idx = random.randint(1, self.length - 1)
        first_child = np.concatenate((self.array[:split_idx], other.array[split_idx:]))
        second_child = np.concatenate((other.array[:split_idx], self.array[split_idx:]))
        return Chromosome(self.length, first_child), Chromosome(self.length, second_child)


def objective_function(*args):
    x1, x2 = args
    return 1.5 - np.exp(-(x1 ** 2) - (x2 ** 2)) - 0.5 * np.exp(-((x1 - 1) ** 2) - ((x2 + 2) ** 2))


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi,
                 population_size=100, tournament_size=2, mutation_probability=0.05,
                 crossover_probability=0.8, num_steps=50):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_length = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = chromosome_length // obj_func_num_args
        self.objective_function = objective_function
        self.aoi = normalize_aoi(aoi, obj_func_num_args)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps
        self.population = [Chromosome(chromosome_length) for _ in range(population_size)]
        self.plot_filename = "ga_trace.png"
        self.plot_title = "Best Individual Trace"

    def decode_chromosome(self, chromosome):
        return [
            chromosome.decode(i * self.bits_per_arg, (i + 1) * self.bits_per_arg, self.aoi[i])
            for i in range(self.obj_func_num_args)
        ]

    def eval_objective_func(self, chromosome):
        return float(self.objective_function(*self.decode_chromosome(chromosome)))

    def tournament_selection(self, fitness_values):
        parents = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(self.population_size), self.tournament_size)
            winner_idx = min(tournament_indices, key=lambda idx: fitness_values[idx])
            parents.append(Chromosome(self.chromosome_length, self.population[winner_idx].array))
        return parents

    def reproduce(self, parents):
        new_population = []
        for i in range(0, self.population_size, 2):
            first_parent = parents[i]
            second_parent = parents[(i + 1) % self.population_size]

            if random.random() < self.crossover_probability:
                first_child, second_child = first_parent.crossover(second_parent)
            else:
                first_child = Chromosome(self.chromosome_length, first_parent.array)
                second_child = Chromosome(self.chromosome_length, second_parent.array)

            first_child.mutation(self.mutation_probability)
            second_child.mutation(self.mutation_probability)
            new_population.extend([first_child, second_child])

        self.population = new_population[:self.population_size]

    def _draw_contour(self, ax, trace, title):
        x_bounds = self.aoi[0]
        y_bounds = self.aoi[1]
        x_values = np.linspace(x_bounds[0], x_bounds[1], 300)
        y_values = np.linspace(y_bounds[0], y_bounds[1], 300)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        grid_z = self.objective_function(grid_x, grid_y)

        trace_x = [point[0] for point in trace]
        trace_y = [point[1] for point in trace]
        point_colors = plt.cm.Reds(np.linspace(0.95, 0.30, len(trace)))

        contour_fill = ax.contourf(grid_x, grid_y, grid_z, levels=30, cmap="viridis")
        ax.contour(grid_x, grid_y, grid_z, levels=15, colors="white", linewidths=0.4, alpha=0.6)
        ax.plot(trace_x, trace_y, color="black", linewidth=1.0, alpha=0.6)
        ax.scatter(trace_x, trace_y, c=point_colors, s=30, edgecolors="black", linewidths=0.25)
        ax.scatter(trace_x[0], trace_y[0], color="#5c0000", marker="s", s=70, label="Start")
        ax.scatter(trace_x[-1], trace_y[-1], color="#ffd6d6", marker="*", s=120,
                   edgecolors="black", linewidths=0.4, label="End")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("$x_1$", fontsize=12)
        ax.set_ylabel("$x_2$", fontsize=12)
        ax.legend(fontsize=9)
        return contour_fill

    def plot_func(self, trace):
        fig, ax = plt.subplots(figsize=(8, 6))
        contour_fill = self._draw_contour(ax, trace, self.plot_title)
        fig.colorbar(contour_fill, ax=ax, label="Objective value")
        fig.tight_layout()
        plt.savefig(self.plot_filename, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def run(self):
        trace = []
        best_value = float("inf")
        best_point = None

        for generation_idx in range(self.num_steps):
            fitness_values = [self.eval_objective_func(chromosome) for chromosome in self.population]
            best_idx = int(np.argmin(fitness_values))
            best_chromosome = self.population[best_idx]
            decoded_values = self.decode_chromosome(best_chromosome)
            current_best_value = fitness_values[best_idx]
            trace.append((decoded_values[0], decoded_values[1], current_best_value))

            if current_best_value < best_value:
                best_value = current_best_value
                best_point = decoded_values

            if generation_idx < self.num_steps - 1:
                parents = self.tournament_selection(fitness_values)
                self.reproduce(parents)

        return {
            "best_point": best_point,
            "best_value": best_value,
            "trace": trace,
        }


def make_study_filename(parameter_name):
    return f"{OUTPUT_PREFIX}_{parameter_name}_study.png"


def run_configuration(seed, ga_kwargs):
    random.seed(seed)
    np.random.seed(seed)
    ga = GeneticAlgorithm(**ga_kwargs)
    return ga, ga.run()


def plot_parameter_study(parameter_name, rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    summary_ax = axes[0, 0]
    values = [row["value"] for row in rows]
    means = [row["mean"] for row in rows]
    stds = [row["std"] for row in rows]
    summary_ax.errorbar(values, means, yerr=stds, fmt="o-", color="#b22222", ecolor="#222222",
                        elinewidth=1.2, capsize=5, markersize=6)
    summary_ax.set_title(f"Effect of {parameter_name.replace('_', ' ')}", fontsize=12)
    summary_ax.set_xlabel(parameter_name.replace("_", " "), fontsize=11)
    summary_ax.set_ylabel("Best objective value", fontsize=11)
    summary_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    colorbar_source = None
    contour_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, row in zip(contour_axes, rows):
        title = f"{parameter_name.replace('_', ' ')} = {row['value']}"
        colorbar_source = row["ga"]._draw_contour(ax, row["trace"], title)

    fig.colorbar(colorbar_source, ax=contour_axes, shrink=0.9, label="Objective value")
    filename = make_study_filename(parameter_name)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return filename


def study_parameter(parameter_name, values, base_kwargs, seeds):
    rows = []

    for value in values:
        ga_kwargs = base_kwargs.copy()
        ga_kwargs[parameter_name] = value
        runs = []

        for seed in seeds:
            ga, result = run_configuration(seed, ga_kwargs)
            runs.append({
                "seed": seed,
                "ga": ga,
                "best_value": result["best_value"],
                "best_point": result["best_point"],
                "trace": result["trace"],
            })

        best_values = np.array([run["best_value"] for run in runs], dtype=float)
        mean_value = float(np.mean(best_values))
        std_value = float(np.std(best_values, ddof=1))
        representative_run = min(runs, key=lambda run: abs(run["best_value"] - mean_value))

        rows.append({
            "value": value,
            "mean": mean_value,
            "std": std_value,
            "representative_seed": representative_run["seed"],
            "representative_best": representative_run["best_value"],
            "representative_point": representative_run["best_point"],
            "ga": representative_run["ga"],
            "trace": representative_run["trace"],
        })

    return rows, plot_parameter_study(parameter_name, rows)


def print_study_results(parameter_name, rows):
    print(f"\n{parameter_name}")
    print("value\tmean_best\tstd_best\trep_seed\trep_best\trep_point")
    for row in rows:
        rep_x1, rep_x2 = row["representative_point"]
        print(
            f"{row['value']}\t{row['mean']:.9f}\t{row['std']:.9f}\t"
            f"{row['representative_seed']}\t{row['representative_best']:.9f}\t"
            f"({rep_x1:.6f}, {rep_x2:.6f})"
        )


def main():
    base_kwargs = {
        "chromosome_length": 32,
        "obj_func_num_args": 2,
        "objective_function": objective_function,
        "aoi": [(-4.0, 4.0), (-4.0, 4.0)],
        "population_size": 60,
        "tournament_size": 4,
        "mutation_probability": 0.05,
        "crossover_probability": 0.8,
        "num_steps": 60,
    }

    parameter_studies = {
        "mutation_probability": [0.0, 0.05, 0.15],
        "tournament_size": [2, 4, 8],
        "population_size": [20, 60, 120],
        "crossover_probability": [0.0, 0.4, 0.8],
    }
    seeds = list(range(8))

    for parameter_name, values in parameter_studies.items():
        rows, summary_filename = study_parameter(parameter_name, values, base_kwargs, seeds)
        print_study_results(parameter_name, rows)
        print(f"summary_plot\t{summary_filename}")


if __name__ == "__main__":
    main()
