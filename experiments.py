"""The experiments as plain data, the learning-rate search, and a runner that can be stopped and restarted.

    python experiments.py depth bias activation --jobs 8 --device cpu
    python experiments.py cnn --device cuda
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import torch

from train import run_name, train_run

EPOCHS = 150
SEEDS = 5
MLP_DEPTHS = (10, 20, 50, 100)
CNN_DEPTHS = (11, 19, 32)
GATE_BIASES = (2.0, 0.0, -1.0, -2.0, -4.0, -10.0)
MLP_GATE_BIAS = -4.0  # the paper's 50-layer CIFAR-100 network (Section 4), confirmed in our pilot
CNN_GATE_BIAS = {11: -1.0, 19: -2.0, 32: -3.0}  # the follow-up's suggestion for these depths (Section 2.2)
CNN_WEIGHT_DECAY = 5e-4  # not given in the papers, chosen in our pilot

# The learning-rate search uses seed 0 and trains on 45,000 training images, keeping the other 5,000 for validation.
LR_GRID = {"mlp": (3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0), "cnn": (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)}
VAL_SIZE = 5000
CNN_SEARCH_EPOCHS = 30  # a full schedule for every CNN and rate would take far too long


def mlp(depth, highway=True, activation="relu", gate_bias=MLP_GATE_BIAS):
    """A fully connected network, without weight decay or augmentation: these are an optimization study."""
    return dict(kind="mlp", depth=depth, highway=highway, activation=activation,
                gate_bias=gate_bias if highway else None, weight_decay=0.0, augment=False)


def cnn(depth, highway=True):
    """A convolutional network, with weight decay and augmentation: these are trained for test accuracy."""
    return dict(kind="cnn", depth=depth, highway=highway, activation="relu", weight_decay=CNN_WEIGHT_DECAY,
                gate_bias=CNN_GATE_BIAS[depth] if highway else None, augment=True)


EXPERIMENTS = {
    "depth": [mlp(depth, highway) for depth in MLP_DEPTHS for highway in (False, True)],
    "cnn": [cnn(depth, highway) for depth in CNN_DEPTHS for highway in (False, True)],
    "bias": [mlp(50, gate_bias=bias) for bias in GATE_BIASES],
    "activation": [mlp(depth, highway, "tanh") for depth in (50, 100) for highway in (False, True)]
                  + [mlp(50, activation="tanh", gate_bias=-2.0)],  # the paper's MNIST setting
}


def run_all(runs, jobs, device):
    """Train the runs, `jobs` at a time. One CPU thread per job, because more don't speed up these small MLPs."""
    with ProcessPoolExecutor(jobs, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        return list(pool.map(train_run, runs, repeat(device)))


def search_run(config, lr, epochs):
    if config["kind"] == "cnn":
        epochs = min(epochs, CNN_SEARCH_EPOCHS)
    return dict(config, lr=lr, seed=0, epochs=epochs, val_size=VAL_SIZE)


def search_score(result):
    """Lower is better: training loss for the MLPs (an optimization study), validation error for the CNNs."""
    if result["diverged"]:
        return float("inf")
    history = result["history"]
    return history["train_loss"][-1] if result["config"]["kind"] == "mlp" else 1 - history["val_acc"][-1]


def find_lrs(configs, epochs, jobs, device):
    """Give every configuration its own learning rate, so that no network loses because of a bad one."""
    runs = [search_run(config, lr, epochs) for config in configs for lr in LR_GRID[config["kind"]]]
    scores = {result["name"]: search_score(result) for result in run_all(runs, jobs, device)}

    def best_lr(config):
        return min(LR_GRID[config["kind"]], key=lambda lr: scores[run_name(search_run(config, lr, epochs))])

    return [best_lr(config) for config in configs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=EXPERIMENTS)
    parser.add_argument("--jobs", type=int, default=1, help="runs at the same time")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    args = parser.parse_args()
    chosen = [config for name in args.experiments for config in EXPERIMENTS[name]]
    configs = [config for i, config in enumerate(chosen) if config not in chosen[:i]]  # experiments share some
    lrs = find_lrs(configs, args.epochs, args.jobs, args.device)
    run_all([dict(config, lr=lr, seed=seed, epochs=args.epochs, val_size=0)
             for config, lr in zip(configs, lrs) for seed in range(args.seeds)], args.jobs, args.device)
