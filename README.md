# EARIN Labs: Introduction to Artificial Intelligence

Laboratory portfolio for the *Introduction to Artificial Intelligence* (EARIN) course at Warsaw University of Technology.
Seven laboratories, each paired with a written report, cover the core areas of the course: classical search, adversarial search, evolutionary optimization, supervised machine learning, neural networks, reinforcement learning and logic programming.

**Group 16, class group 102:** Leonardo Meloni and Anatoli Abenet Abahumna.

## Topics Covered

| Lab | Topic | AI area | Language | Folder |
| --- | --- | --- | --- | --- |
| 1 | Maze solving with BFS and DFS | Uninformed search | Python | [`lab1-maze-search`](lab1-maze-search) |
| 2 | Tic-Tac-Toe against an optimal opponent | Adversarial search (Minimax, Alpha-Beta) | Python | [`lab2-tictactoe-minimax`](lab2-tictactoe-minimax) |
| 3 | Global minimization of a 2D function | Evolutionary computation (Genetic Algorithm) | Python | [`lab3-genetic-algorithm`](lab3-genetic-algorithm) |
| 4 | House price prediction on California Housing | Supervised learning (regression) | Python | [`lab4-california-housing-regression`](lab4-california-housing-regression) |
| 5 | Multilayer Perceptron from scratch on MNIST | Neural networks | Python (NumPy) | [`lab5-mlp-mnist`](lab5-mlp-mnist) |
| 6 | CartPole balancing with tabular Q-Learning | Reinforcement learning | Python (Gymnasium) | [`lab6-cartpole-q-learning`](lab6-cartpole-q-learning) |
| 7 | Date arithmetic on a leap year | Logic programming | Prolog | [`lab7-prolog-date-arithmetic`](lab7-prolog-date-arithmetic) |

Each lab folder contains the solution script and the PDF report (`labN_cg102_g16_..._Meloni_Abahumna.pdf`), which describes the task, the algorithm, the implementation and the results in detail.

---

## Lab 1: Maze Solving with BFS and DFS

A maze is given as a 2D grid (`0` empty cell, `1` wall), together with a start and a finish position. The program returns the number of steps from start to finish, or `-1` if no path exists, and visualizes the whole search, including every visited cell and not only the final path.

- **Breadth-First Search** explores the maze level by level and, on an unweighted grid, always finds the shortest path.
- **Depth-First Search** follows one branch as deep as possible before backtracking; it does not guarantee the shortest path and depends on the neighbour exploration order.
- Input validation protects the solver against empty, non-rectangular or out-of-range inputs.
- The required interface is `bfs(maze, start, finish)`, `dfs(maze, start, finish)` and `vizualize(viz)`.

## Lab 2: Tic-Tac-Toe with Minimax and Alpha-Beta Pruning

An interactive game on a 3×3 board where a human plays against the computer, who chooses moves optimally. The user decides whether to play first or second, inputs are validated, the board is printed after every move, and wins and draws are detected.

- **Minimax** builds the game tree assuming perfect play from both sides.
- **Alpha-Beta pruning** discards branches that cannot affect the decision, reducing the number of evaluated nodes without changing the result.
- The evaluation function takes the remaining depth into account, so the computer prefers immediate wins and delays defeats.
- The solution is encapsulated in a single `TicTacToe` class.

## Lab 3: Genetic Algorithm

A genetic algorithm searches for the global minimum of a two-dimensional function with a complex landscape, in the interval [-5, 5] for both variables. The landscape contains a global minimum and a shallower local minimum that acts as a trap.

- **Representation:** a 32-bit binary chromosome (16 bits per variable) decoded into the continuous search space.
- **Operators:** tournament selection, crossover and mutation.
- **Configuration:** population of 100, tournament size of 3, 50 generations.
- **Parameter study:** the report analyses how population size and the other operators affect convergence, and highlights the balance between exploration and exploitation. A drastic reduction of the population size compromises the diversity needed to explore the search space.

## Lab 4: Regression on California Housing

Prediction of the median house value of a census block group using the California Housing dataset from scikit-learn (20,640 observations, 8 numerical features, 80/20 split).

- **Data analysis:** correlation matrix and feature analysis; all eight features were kept, since none is linearly dependent. The target is capped at 5.0, a property of the dataset the models have to cope with.
- **Preprocessing:** `StandardScaler` only inside the Ridge pipeline; tree-based models run on raw features, as they are invariant to monotonic scaling.
- **Models and tuning:** Ridge Regression, Random Forest and XGBoost, tuned with grid search and 4-fold cross-validation.

| Model | Test MAE | Test RMSE | Test R² | Best CV RMSE |
| --- | ---: | ---: | ---: | ---: |
| Ridge Regression | 0.5332 | 0.7456 | 0.5758 | 0.7214 |
| Random Forest | 0.3268 | 0.5040 | 0.8062 | 0.5134 |
| **XGBoost** | **0.2966** | **0.4566** | **0.8409** | **0.4645** |

Target values are expressed in hundreds of thousands of dollars. Results, plots and the comparison tables are in `Results/`.

## Lab 5: Multilayer Perceptron from Scratch

A multilayer perceptron implemented with NumPy only (layers, activations, loss, forward pass and backpropagation) and trained on MNIST with SGD and MSE loss. The lab compares four hidden-layer activation functions under identical conditions: Tanh, Sigmoid, ReLU and LeakyReLU (α = 0.01).

- **Setup:** 3 random seeds, batch size 128, 40 epochs, learning rate 0.1.
- **Activation comparison** (baseline 64-32 architecture, mean test accuracy): LeakyReLU 97.29%, ReLU 97.20%, Tanh 97.13%, Sigmoid 93.26%.
- **Architecture comparison** with the best activation: 128-64 reached 97.61%, a deeper network 97.49% and the 64-32 baseline 97.29%.
- **Discussion:** the slow convergence of Sigmoid illustrates the vanishing gradient problem, while ReLU and LeakyReLU keep a constant gradient for positive inputs; LeakyReLU additionally avoids dead neurons.
- `tests_results/` contains the exploratory runs with other epoch and learning-rate settings.

## Lab 6: Q-Learning on CartPole

Tabular Q-Learning on the `CartPole-v1` environment from Gymnasium. Since the observations are continuous, each dimension is clipped and discretized into bins to index the Q-table (6 × 8 × 12 × 12 × 2 = 13,824 values with the default setting).

- **Training:** epsilon-greedy exploration, 5,000 episodes, α = 0.1, γ = 0.99, ε decaying from 1.0 to 0.05, seed 42; evaluation with greedy actions.
- **Experiments:** one-factor-at-a-time sweeps over the learning rate, discount factor, initial exploration rate, exploration decay and quantization size.
- **Baseline result:** greedy evaluation average of 251.9 steps (maximum 500).
- **Outputs:** per-run metrics, configuration and evaluation files, plus reward, episode-length and epsilon plots and comparison figures.

## Lab 7: Date Arithmetic in Prolog

Addition and subtraction of a number of days (0 to 366) to a date written as a `"DDMM"` string, for the leap year 2024. Dates are converted to an ordinal day of the year, so the arithmetic reduces to integer operations, and the result is converted back; results that leave 2024 fail instead of wrapping around.

- **Prolog features used:** facts (month lengths), rules, unification, recursion, pattern matching in predicate heads and backtracking.
- **Predicates:** `add_date(+DateStr, +N)` and `sub_date(+DateStr, +N)`.

---

## Repository Structure

```text
.
├── docs/                              Course and report instructions
├── lab1-maze-search/                  BFS and DFS
├── lab2-tictactoe-minimax/            Minimax with Alpha-Beta pruning
├── lab3-genetic-algorithm/            Genetic algorithm
├── lab4-california-housing-regression/
│   ├── Dataset/                       Train and test splits
│   ├── Models/                        Grid search results and saved models
│   ├── Results/                       Plots and comparison tables
│   └── PDF Report/                    Report
├── lab5-mlp-mnist/
│   └── tests_results/                 Activation and architecture experiments
├── lab6-cartpole-q-learning/
│   ├── results/                       Per-run data and figures
│   └── report/                        Report and selected figures
├── lab7-prolog-date-arithmetic/       Prolog solution
├── requirements.txt                   Python dependencies for all labs
└── README.md
```

## Setup

The Python labs require Python 3.10 or higher.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Labs 1, 2 and 7 need no external packages. Lab 7 requires [SWI-Prolog](https://www.swi-prolog.org/) or the online [SWISH](https://swish.swi-prolog.org/) environment.

## Running the Labs

Run each command from inside the corresponding lab folder.

```bash
# Lab 1: maze solver (runs the examples and custom tests)
python lab1_cg102_g16_v1_Meloni_Abahumna.py

# Lab 2: interactive Tic-Tac-Toe
python lab2_cg102_g16_v1_Meloni_Abahumna.py

# Lab 3: genetic algorithm
python lab3_cg102_g16_v1_Meloni_Abahumna.py

# Lab 4: regression pipeline
python lab4_cg102_g16_v1_Meloni_Abahumna.py

# Lab 5: MLP on MNIST (downloads MNIST on first run)
python lab5_cg102_g16_vA_Meloni_Abahumna.py

# Lab 6: single baseline training, or the full set of experiments
python lab6_cg102_g16_vA_Meloni_Abahumna.py
python lab6_cg102_g16_vA_Meloni_Abahumna.py --run-experiments
```

Lab 7, from SWI-Prolog:

```prolog
?- [lab7_cg102_g16_vA_Meloni_Abahumna].
?- add_date("2205", 14).
?- sub_date("2205", 10).
```

Lab 4 and Lab 6 have their own `README.md` with additional options.

## Authors

Leonardo Meloni and Anatoli Abenet Abahumna, Warsaw University of Technology, Group 16.
