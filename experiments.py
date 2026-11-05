"""The experiments as plain data, the learning-rate search, and a runner that can be stopped and restarted.

    python experiments.py depth bias activation --jobs 8 --device cpu
    python experiments.py cnn --device cuda
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import torch

from models import CNN_LAYOUTS
from train import run_name, train_run

EPOCHS, SEEDS, CNN_SEARCH_EPOCHS, VAL_SIZE = 150, 5, 30, 5000
GRID = {"mlp": (3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0), "cnn": (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)}
BIASES = (2.0, 0.0, -1.0, -2.0, -4.0, -10.0)
MLP_GATE_BIAS = -4.0  # the paper's 50-layer CIFAR-100 network (Section 4), confirmed in our pilot
CNN_WEIGHT_DECAY = 5e-4  # not given in the papers, chosen in our pilot


def mlp(depth, highway=True, activation="relu", gate_bias=MLP_GATE_BIAS):
    return dict(kind="mlp", depth=depth, highway=highway, activation=activation,
                gate_bias=gate_bias if highway else None, weight_decay=0.0, augment=False)


def cnn(depth, highway):
    return dict(kind="cnn", depth=depth, highway=highway, activation="relu", weight_decay=CNN_WEIGHT_DECAY,
                gate_bias=CNN_LAYOUTS[depth]["gate_bias"] if highway else None, augment=True)


EXPERIMENTS = {
    "depth": [mlp(d, h) for d in (10, 20, 50, 100) for h in (False, True)],
    "cnn": [cnn(d, h) for d in (11, 19, 32) for h in (False, True)],
    "bias": [mlp(50, gate_bias=b) for b in BIASES],
    "activation": [mlp(d, h, "tanh") for d in (50, 100) for h in (False, True)] + [mlp(50, True, "tanh", -2.0)],
}


def run_all(runs, args):
    """Train the runs, args.jobs at a time. One CPU thread per job, because more don't speed up these small MLPs."""
    with ProcessPoolExecutor(args.jobs, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        return list(pool.map(train_run, runs, repeat(args.device)))


def search_run(config, lr, args):
    """Seed 0 on a 45K/5K split. CNNs get a shorter schedule, because a full one would take far too long."""
    epochs = args.epochs if config["kind"] == "mlp" else min(args.epochs, CNN_SEARCH_EPOCHS)
    return dict(config, lr=lr, seed=0, epochs=epochs, val_size=VAL_SIZE)


def search_score(result):
    """Lower is better: training loss for the MLPs (an optimization study), validation error for the CNNs."""
    history = result["history"]
    score = history["train_loss"][-1] if result["config"]["kind"] == "mlp" else 1 - history["val_acc"][-1]
    return float("inf") if result["diverged"] else score


def find_lrs(configs, args):
    """Give every configuration its own learning rate, so that no network loses because of a bad one."""
    runs = [search_run(config, lr, args) for config in configs for lr in GRID[config["kind"]]]
    scores = {result["name"]: search_score(result) for result in run_all(runs, args)}
    return [min(GRID[config["kind"]], key=lambda lr: scores[run_name(search_run(config, lr, args))])
            for config in configs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=EXPERIMENTS)
    parser.add_argument("--jobs", type=int, default=1, help="runs at the same time")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    args = parser.parse_args()
    # Experiments share identical configurations, and each one is trained only once.
    configs = list({str(config): config for name in args.experiments for config in EXPERIMENTS[name]}.values())
    run_all([dict(config, lr=lr, seed=seed, epochs=args.epochs, val_size=0)
             for config, lr in zip(configs, find_lrs(configs, args)) for seed in range(args.seeds)], args)
