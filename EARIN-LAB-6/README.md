# Lab 6 - CartPole Q-Learning

This project implements tabular Q-Learning for **Variant 1: CartPole-v1**. Since the CartPole observations are continuous, the script clips each observation dimension and maps it into discrete bins before indexing the Q-table.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run One Baseline Training

```bash
python3 lab6_cg102_g16_vA_Meloni_Abahumna.py
```

Useful options:

```bash
python3 lab6_cg102_g16_vA_Meloni_Abahumna.py --help
python3 lab6_cg102_g16_vA_Meloni_Abahumna.py --episodes 5000 --alpha 0.1 --gamma 0.99 --bins 6,8,12,12
```

## Run All Experiments

```bash
python3 lab6_cg102_g16_vA_Meloni_Abahumna.py --run-experiments
```

This runs the baseline plus one-factor-at-a-time sweeps for learning rate, discount factor, initial exploration rate, exploration decay, and quantization size.

## Outputs

- `results/data/*_metrics.csv`: reward, episode length, epsilon, and rolling average per episode.
- `results/data/*_config.json`: full hyperparameters for each run.
- `results/data/*_evaluation.json`: greedy evaluation rewards.
- `results/data/experiment_summary.csv`: final table for all sweeps.
- `results/figures/*.png`: reward, episode length, epsilon, and comparison plots.
- `report/figures/*.png`: selected plots copied for the report.
- `report/report.pdf`: final report for submission.
