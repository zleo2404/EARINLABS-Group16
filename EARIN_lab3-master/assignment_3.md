# Assignment 3 – Optimization with a Genetic Algorithm (5 points)

## Common Instructions

The goal of this assignment is to solve an optimization problem using a **genetic algorithm**. You must find the **minimum** of the objective function assigned to your group using **point crossover**, **mutation**, and **tournament selection**.

Use the provided code skeleton (`template.py`) and implement all required methods. The implementation must be **universal** — the dimension of the function can be arbitrary (use `*args`).

---

## Skeleton Overview

### `Chromosome` class

| Method | Description |
|--------|-------------|
| `__init__(length, array=None)` | Initialize with a binary vector of the given length. If `array` is `None`, generate a random binary vector. |
| `decode(lower_bound, upper_bound, aoi)` | Decode the chromosome segment from bit `lower_bound` to bit `upper_bound` into a real number within range `aoi`. You may use the provided `min_max_norm` helper. |
| `mutation(probability)` | With the given probability, flip the value of one randomly selected gene. |
| `crossover(other)` | Perform one-point crossover with another chromosome. Return two offspring. |

### `GeneticAlgorithm` class

| Method | Description |
|--------|-------------|
| `eval_objective_func(chromosome)` | Decode the full chromosome into function arguments and return the objective function value. |
| `tournament_selection()` | Run tournament selection over the current population. Return selected parents. |
| `reproduce(parents)` | Generate a new population. With probability `crossover_probability`, perform crossover; otherwise pass parents directly to the next generation. |
| `run()` | Execute the full GA for `num_steps` generations. In each generation, record the decoded variable values and objective function value for the best individual. |

---

## Scoring (5 points)

| Component | Points |
|-----------|--------|
| **Code** — correct implementation of all required methods | 1.0 |
| **Report** — plot, analysis, and written conclusions | 2.0 |
| **Discussion** — oral discussion of the solution with the instructor | 2.0 |
| **Total** | **5.0** |

> **Note:** If during the oral discussion it becomes apparent that the student does not understand their own solution, the instructor reserves the right to award **0 points** for the entire assignment.

---

## Decoding Example

A binary string `10000111` (8 bits) represents the decimal value **135**.
If `aoi = [0, 1]`, then:

```
decoded = min_max_norm(135, 0, 2^8 - 1, 0, 1) ≈ 0.529
```

More generally, a segment of `k` bits encodes integers from `0` to `2^k − 1`, which are normalized to the target range via `min_max_norm`.

---

## Plot

Use the `plot_func(trace)` method to generate a **contour plot** of your objective function. Each point corresponds to the best individual in that generation. Points should be colored from **dark to light red** as generations progress toward the minimum.

---

## Parameter Study

You are required to **systematically test the effect of key hyperparameters** on algorithm performance and support your findings with visualizations. For each parameter, run multiple experiments varying only that parameter while keeping all others fixed.

### Parameters to test

| Parameter |
|-----------|
| `mutation_probability` |
| `tournament_size` |
| `population_size` |
| `crossover_probability` |


### Required Visualizations and Analysis

Each configuration should be run multiple times (e.g. 5–10) due to the randomness of the initial population. For each configuration, report the mean and standard deviation of the found minima. For each of the investigated parameter values, include a single contour plot with best-individual trace (from `plot_func`) — one plot per configuration, showing the path taken through the search space. Provide conclusions.

---

# Group A

## Objective Function

$$f(x_1, x_2) = 1.5 - e^{-x_1^2 - x_2^2} - 0.5\,e^{-(x_1-1)^2-(x_2+2)^2}$$

---

# Group B

## Objective Function

$$f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2$$

This is known as **Himmelblau's function**.

## Notes

- Himmelblau's function has **four global minima** of equal value. The algorithm may converge to any of them — this is expected.

---

# Group C

## Objective Function

$$f(x_1, x_2) = 20 + x_1^2 - 10\cos(2\pi x_1) + x_2^2 - 10\cos(2\pi x_2)$$

This is known as the **Rastrigin function**.

## Notes

- The Rastrigin function has a large number of regularly distributed local minima, making it one of the hardest functions for population-based optimizers.
- Despite the many local traps, the global minimum at the origin is clearly the deepest.

---

# Group D

## Objective Function

$$f(x_1, x_2) = \frac{1}{2}\left(x_1^4 - 16x_1^2 + 5x_1 + x_2^4 - 16x_2^2 + 5x_2\right)$$

This is known as the **Styblinski-Tang function**.

## Notes

- The Styblinski-Tang function has several clearly separated local minima visible in both the contour plot and the 3D surface, making it well-suited for studying how the algorithm navigates between competing basins.

---

# Group E

## Objective Function

$$f(x_1, x_2) = \left(4 - 2.1\,x_1^2 + \frac{x_1^4}{3}\right)x_1^2 + x_1 x_2 + \left(-4 + 4\,x_2^2\right)x_2^2$$

This is known as the **six-hump camelback function**.

## Notes

- The function has **six distinct humps** and **two symmetric global minima**, making it a challenging multimodal landscape.
- The two global minima are symmetric about the origin — the algorithm may converge to either one.

---

## Technical Details

- The submission should include the python file and the report in the format of `.pdf`
- Please ensure that your code adheres to basic standards of lean coding in accordance with PEP8. Additionally, it should contain comments on the crucial parts to help with readability and understanding.
- The inputs to the code should be provided inside the python code or from the file. In case of providing inputs from the file, the python code should include reading from the file part and provide instructions in the comments. In addition, the file with the inputs should be included to the submission.
- All the test cases and examples used should be in the submission. The code without proper test cases may get less points.
- Templates contain crucial information for some tasks (e.g. function definition), so, start from them. However, they can be changed and adapted as long as objectives of the task are reached.

## Submission Guidelines

You should submit all the files to private GitHub repository not later than:

- **18.04.2026 until 23:59** for Monday group
- **20.04.2026 until 23:59** for Wednesday group
- **22.04.2026 until 23:59** for Friday group

The on-line assessment will take place during your labs in two weeks. In case of questions, please contact me via Teams.

Names of all files should be:

```
lab3_cg{class_group_number}_g{group_number}_v{variant_number}_{surname1}_{surname2}.(py,pdf,zip, etc)
```

E.g. `lab3_cg101_g1_v1_Smith_Jankowski.py`