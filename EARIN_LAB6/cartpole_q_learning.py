"""Tabular Q-Learning for CartPole-v1 with observation quantization.

This file is intentionally self-contained for the laboratory submission:
it defines the configuration, helper functions, training/evaluation code,
plotting, and command line interface in one place.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cartpole_q_learning_matplotlib")
)

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENV_NAME = "CartPole-v1"
OBSERVATION_LOW = np.array([-4.8, -3.0, -0.418, -3.5], dtype=np.float64)
OBSERVATION_HIGH = np.array([4.8, 3.0, 0.418, 3.5], dtype=np.float64)
DEFAULT_BINS = (6, 8, 12, 12)


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 10000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    bins: Tuple[int, int, int, int] = DEFAULT_BINS
    seed: int = 42
    rolling_window: int = 100
    solved_reward: float = 475.0
    stop_when_solved: bool = False
    env_name: str = ENV_NAME


def discretize_observation(
    observation: Sequence[float], bins: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Convert a continuous CartPole observation to a discrete Q-table state."""
    clipped = np.clip(np.asarray(observation, dtype=np.float64), OBSERVATION_LOW, OBSERVATION_HIGH)
    ratios = (clipped - OBSERVATION_LOW) / (OBSERVATION_HIGH - OBSERVATION_LOW)
    state = np.floor(ratios * np.asarray(bins)).astype(int)
    state = np.clip(state, 0, np.asarray(bins) - 1)
    return tuple(int(value) for value in state)


def choose_action(
    q_table: np.ndarray,
    state: Tuple[int, int, int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Choose an action with an epsilon-greedy policy."""
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[-1]))

    q_values = q_table[state]
    best_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(best_actions))


def update_q_value(
    q_table: np.ndarray,
    state: Tuple[int, int, int, int],
    action: int,
    reward: float,
    next_state: Tuple[int, int, int, int],
    done: bool,
    alpha: float,
    gamma: float,
) -> None:
    """Apply the standard one-step Q-Learning update."""
    old_value = q_table[state + (action,)]
    next_best = 0.0 if done else float(np.max(q_table[next_state]))
    target = reward + gamma * next_best
    q_table[state + (action,)] = old_value + alpha * (target - old_value)


def train(config: TrainingConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    """Train a Q-table and return it together with episode metrics."""
    rng = np.random.default_rng(config.seed)
    env = gym.make(config.env_name)
    env.action_space.seed(config.seed)

    q_shape = tuple(config.bins) + (env.action_space.n,)
    q_table = np.zeros(q_shape, dtype=np.float64)

    rewards: List[float] = []
    lengths: List[int] = []
    epsilons: List[float] = []
    rolling_averages: List[float] = []
    solved_episode: Optional[int] = None
    epsilon = config.epsilon_start

    for episode in range(1, config.episodes + 1):
        reset_seed = config.seed + episode
        observation, _ = env.reset(seed=reset_seed)
        state = discretize_observation(observation, config.bins)

        total_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = choose_action(q_table, state, epsilon, rng)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_observation(next_observation, config.bins)
            done = terminated or truncated

            update_q_value(
                q_table=q_table,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                alpha=config.alpha,
                gamma=config.gamma,
            )

            state = next_state
            total_reward += float(reward)
            episode_length += 1

        rewards.append(total_reward)
        lengths.append(episode_length)
        epsilons.append(epsilon)

        recent_rewards = rewards[-config.rolling_window :]
        rolling_average = float(np.mean(recent_rewards))
        rolling_averages.append(rolling_average)

        enough_history = len(rewards) >= config.rolling_window
        if enough_history and solved_episode is None and rolling_average >= config.solved_reward:
            solved_episode = episode
            if config.stop_when_solved:
                break

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

    env.close()
    metrics: Dict[str, object] = {
        "total_rewards": rewards,
        "episode_lengths": lengths,
        "epsilons": epsilons,
        "rolling_average_rewards": rolling_averages,
        "solved_episode": solved_episode,
    }
    return q_table, metrics


def evaluate(
    q_table: np.ndarray, config: TrainingConfig, episodes: int = 20
) -> List[float]:
    """Evaluate the learned Q-table using greedy actions."""
    env = gym.make(config.env_name)
    rewards: List[float] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=config.seed + 100_000 + episode)
        state = discretize_observation(observation, config.bins)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(np.argmax(q_table[state]))
            observation, reward, terminated, truncated, _ = env.step(action)
            state = discretize_observation(observation, config.bins)
            total_reward += float(reward)

        rewards.append(total_reward)

    env.close()
    return rewards


def plot_metrics(
    metrics: Dict[str, object], output_dir: Path, run_name: str
) -> Dict[str, Path]:
    """Save reward, episode length, and epsilon plots for one training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(metrics["total_rewards"]) + 1)
    rewards = np.asarray(metrics["total_rewards"], dtype=np.float64)
    lengths = np.asarray(metrics["episode_lengths"], dtype=np.float64)
    epsilons = np.asarray(metrics["epsilons"], dtype=np.float64)
    rolling = np.asarray(metrics["rolling_average_rewards"], dtype=np.float64)

    paths = {
        "rewards": output_dir / f"{run_name}_rewards.png",
        "lengths": output_dir / f"{run_name}_lengths.png",
        "epsilon": output_dir / f"{run_name}_epsilon.png",
    }

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color="#6aa4d8", alpha=0.35, linewidth=0.8, label="Episode reward")
    plt.plot(episodes, rolling, color="#1f4e79", linewidth=2.0, label="Rolling average (100)")
    plt.axhline(475.0, color="#b23b3b", linestyle="--", linewidth=1.2, label="Solved threshold")
    plt.title(f"Total reward - {run_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["rewards"], dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, lengths, color="#27706b", linewidth=1.0)
    plt.title(f"Episode length - {run_name}")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(paths["lengths"], dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons, color="#7c4d1d", linewidth=1.6)
    plt.title(f"Epsilon - {run_name}")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(paths["epsilon"], dpi=160)
    plt.close()

    return paths


def plot_sweep_comparison(
    sweep_name: str,
    runs: Sequence[Tuple[str, str, Dict[str, object]]],
    output_dir: Path,
) -> Path:
    """Save one comparison plot showing rolling reward curves for a sweep."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"comparison_{sweep_name}.png"

    plt.figure(figsize=(10, 5))
    for setting_label, run_name, metrics in runs:
        episodes = np.arange(1, len(metrics["rolling_average_rewards"]) + 1)
        rolling = np.asarray(metrics["rolling_average_rewards"], dtype=np.float64)
        plt.plot(episodes, rolling, linewidth=1.6, label=setting_label)

    plt.axhline(475.0, color="#b23b3b", linestyle="--", linewidth=1.2, label="Solved threshold")
    plt.title(f"Rolling average reward comparison - {sweep_name}")
    plt.xlabel("Episode")
    plt.ylabel("Rolling average reward")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def parse_bins(value: str) -> Tuple[int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("bins must be four positive integers, for example 6,8,12,12")
    return parts


def safe_value(value: object) -> str:
    if isinstance(value, tuple):
        return "x".join(str(part) for part in value)
    return str(value).replace(".", "p").replace("-", "m")


def format_value(value: object) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(str(part) for part in value) + ")"
    return str(value)


def save_metrics_csv(metrics: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "total_reward", "episode_length", "epsilon", "rolling_average_reward"])
        rows = zip(
            metrics["total_rewards"],
            metrics["episode_lengths"],
            metrics["epsilons"],
            metrics["rolling_average_rewards"],
        )
        for index, row in enumerate(rows, start=1):
            writer.writerow([index, *row])


def save_summary_csv(summaries: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_name",
        "sweep",
        "setting",
        "episodes",
        "alpha",
        "gamma",
        "epsilon_start",
        "epsilon_min",
        "epsilon_decay",
        "bins",
        "seed",
        "final_rolling_reward",
        "best_rolling_reward",
        "solved_episode",
        "evaluation_average",
        "evaluation_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({column: summary.get(column, "") for column in columns})


def copy_for_report(paths: Iterable[Path], report_figures_dir: Path) -> None:
    report_figures_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, report_figures_dir / path.name)


def make_summary(
    run_name: str,
    sweep: str,
    setting: str,
    config: TrainingConfig,
    metrics: Dict[str, object],
    evaluation_rewards: Sequence[float],
) -> Dict[str, object]:
    rolling = np.asarray(metrics["rolling_average_rewards"], dtype=np.float64)
    summary = {
        "run_name": run_name,
        "sweep": sweep,
        "setting": setting,
        "episodes": len(metrics["total_rewards"]),
        "alpha": config.alpha,
        "gamma": config.gamma,
        "epsilon_start": config.epsilon_start,
        "epsilon_min": config.epsilon_min,
        "epsilon_decay": config.epsilon_decay,
        "bins": "x".join(str(part) for part in config.bins),
        "seed": config.seed,
        "final_rolling_reward": float(rolling[-1]),
        "best_rolling_reward": float(np.max(rolling)),
        "solved_episode": metrics["solved_episode"] if metrics["solved_episode"] is not None else "",
        "evaluation_average": float(np.mean(evaluation_rewards)),
        "evaluation_std": float(np.std(evaluation_rewards)),
    }
    return summary


def run_single_training(
    config: TrainingConfig,
    run_name: str,
    sweep: str,
    setting: str,
    results_dir: Path,
    evaluate_episodes: int,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, Path]]:
    q_table, metrics = train(config)
    evaluation_rewards = evaluate(q_table, config, episodes=evaluate_episodes)
    summary = make_summary(run_name, sweep, setting, config, metrics, evaluation_rewards)

    data_dir = results_dir / "data"
    figure_dir = results_dir / "figures"
    save_metrics_csv(metrics, data_dir / f"{run_name}_metrics.csv")
    with (data_dir / f"{run_name}_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(asdict(config), config_file, indent=2)
    with (data_dir / f"{run_name}_evaluation.json").open("w", encoding="utf-8") as evaluation_file:
        json.dump({"rewards": evaluation_rewards}, evaluation_file, indent=2)

    plot_paths = plot_metrics(metrics, figure_dir, run_name)
    return summary, metrics, plot_paths


def experiment_configs(base_config: TrainingConfig) -> List[Tuple[str, str, str, TrainingConfig]]:
    experiments: List[Tuple[str, str, str, TrainingConfig]] = [
        ("baseline", "baseline", "default", base_config)
    ]

    for alpha in (0.05, 0.1, 0.2):
        experiments.append(
            (f"learning_rate_alpha_{safe_value(alpha)}", "learning_rate", f"alpha = {alpha}", replace(base_config, alpha=alpha))
        )

    for gamma in (0.95, 0.99, 0.999):
        experiments.append(
            (f"discount_factor_gamma_{safe_value(gamma)}", "discount_factor", f"gamma = {gamma}", replace(base_config, gamma=gamma))
        )

    for epsilon_start in (0.5, 0.8, 1.0):
        experiments.append(
            (
                f"exploration_start_{safe_value(epsilon_start)}",
                "exploration_start",
                f"epsilon_start = {epsilon_start}",
                replace(base_config, epsilon_start=epsilon_start),
            )
        )

    for epsilon_decay in (0.99, 0.995, 0.999):
        experiments.append(
            (
                f"exploration_decay_{safe_value(epsilon_decay)}",
                "exploration_decay",
                f"epsilon_decay = {epsilon_decay}",
                replace(base_config, epsilon_decay=epsilon_decay),
            )
        )

    for bins in ((4, 6, 8, 8), (6, 8, 12, 12), (8, 10, 14, 14)):
        experiments.append(
            (
                f"quantization_bins_{safe_value(bins)}",
                "quantization",
                f"bins = {format_value(bins)}",
                replace(base_config, bins=bins),
            )
        )

    return experiments


def run_experiments(
    base_config: TrainingConfig,
    results_dir: Path,
    report_figures_dir: Path,
    evaluate_episodes: int,
) -> None:
    summaries: List[Dict[str, object]] = []
    comparison_runs: Dict[str, List[Tuple[str, str, Dict[str, object]]]] = {}
    report_paths: List[Path] = []

    for index, (run_name, sweep, setting, config) in enumerate(experiment_configs(base_config), start=1):
        print(f"[{index:02d}] Training {run_name} ({setting})")
        summary, metrics, plot_paths = run_single_training(
            config=config,
            run_name=run_name,
            sweep=sweep,
            setting=setting,
            results_dir=results_dir,
            evaluate_episodes=evaluate_episodes,
        )
        summaries.append(summary)
        comparison_runs.setdefault(sweep, []).append((setting, run_name, metrics))

        if run_name == "baseline":
            report_paths.extend(plot_paths.values())

        solved_text = summary["solved_episode"] if summary["solved_episode"] != "" else "not reached"
        print(
            "    final rolling = "
            f"{summary['final_rolling_reward']:.2f}, best rolling = {summary['best_rolling_reward']:.2f}, "
            f"solved episode = {solved_text}, eval avg = {summary['evaluation_average']:.2f}"
        )

    for sweep, runs in comparison_runs.items():
        if sweep == "baseline":
            continue
        report_paths.append(plot_sweep_comparison(sweep, runs, results_dir / "figures"))

    save_summary_csv(summaries, results_dir / "data" / "experiment_summary.csv")
    copy_for_report(report_paths, report_figures_dir)
    print(f"Saved summary table to {results_dir / 'data' / 'experiment_summary.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train tabular Q-Learning on CartPole-v1 using quantized observations."
    )
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon value.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon value.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode epsilon multiplier.")
    parser.add_argument("--bins", type=parse_bins, default=DEFAULT_BINS, help="Observation bins, for example 6,8,12,12.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-name", default="baseline", help="Name used for output files in single-run mode.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for result files.")
    parser.add_argument(
        "--report-figures-dir",
        type=Path,
        default=Path("report") / "figures",
        help="Directory where selected figures are copied for the report.",
    )
    parser.add_argument("--evaluate-episodes", type=int, default=20, help="Greedy evaluation episodes after training.")
    parser.add_argument("--run-experiments", action="store_true", help="Run baseline and all requested sweeps.")
    parser.add_argument("--stop-when-solved", action="store_true", help="Stop a run when the solved threshold is reached.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TrainingConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        bins=args.bins,
        seed=args.seed,
        stop_when_solved=args.stop_when_solved,
    )

    if args.run_experiments:
        run_experiments(
            base_config=config,
            results_dir=args.results_dir,
            report_figures_dir=args.report_figures_dir,
            evaluate_episodes=args.evaluate_episodes,
        )
        return

    summary, _metrics, plot_paths = run_single_training(
        config=config,
        run_name=args.run_name,
        sweep="single",
        setting="custom",
        results_dir=args.results_dir,
        evaluate_episodes=args.evaluate_episodes,
    )
    save_summary_csv([summary], args.results_dir / "data" / f"{args.run_name}_summary.csv")
    copy_for_report(plot_paths.values(), args.report_figures_dir)

    solved_text = summary["solved_episode"] if summary["solved_episode"] != "" else "not reached"
    print(f"Run name: {args.run_name}")
    print(f"Final rolling reward: {summary['final_rolling_reward']:.2f}")
    print(f"Best rolling reward: {summary['best_rolling_reward']:.2f}")
    print(f"Solved episode: {solved_text}")
    print(f"Evaluation average over {args.evaluate_episodes} episodes: {summary['evaluation_average']:.2f}")


if __name__ == "__main__":
    main()
