"""Everything the report shows, computed from saved results and weights. A CPU is enough.

    python analyze.py                # the table and all figures
    python analyze.py table curves   # only some of them
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from data import load_cifar10
from experiments import BIASES, cnn, mlp
from models import ACTIVATIONS, Highway, build_model
from train import evaluate

plt.switch_backend("Agg")
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "lines.linewidth": 1.4, "legend.frameon": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "savefig.dpi": 200, "savefig.bbox": "tight", "figure.constrained_layout.use": True})
PLAIN, HIGHWAY, MUTED = "#eb6834", "#2a78d6", "#898781"
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
CLASSES = {0: "airplane", 1: "automobile", 3: "cat", 8: "ship"}


def key(config):
    return config["kind"], config["depth"], config["highway"], config["activation"], config["gate_bias"]


def load_runs():
    """The final runs that did not diverge, grouped by configuration, with one result per seed."""
    runs = {}
    for path in sorted(Path("results").glob("*.json")):
        run = json.loads(path.read_text())
        if not run["diverged"]:
            runs.setdefault(key(run["config"]), []).append(run)
    return runs


def load_model(run, device):
    saved = torch.load(Path("checkpoints") / f"{run['name']}.pt", map_location=device, weights_only=True)
    model = build_model(saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return model, [module for module in model if isinstance(module, Highway)]


def best_model(runs, config, device):
    """The seed with the lowest training loss: the paper also inspects its best network (Section 4)."""
    return load_model(min(runs[key(config)], key=lambda run: run["history"]["train_loss"][-1]), device)


def table(runs, device):
    """Re-evaluate the saved weights and report them in the follow-up's format: best (mean ± std)."""
    test = load_cifar10(device)["test"]
    lines = ["| Network | Layers | Activation | Gate bias | Parameters | Learning rate | Training loss "
             "| Test accuracy (%) | Seeds finished |", "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(runs.items(), key=lambda item: (*item[0][:4], item[0][4] or 0))
    for (kind, depth, highway, activation, bias), group in order:
        acc = np.array([100 * evaluate(load_model(run, device)[0], *test)[1] for run in group])
        loss = np.median([run["history"]["train_loss"][-1] for run in group])
        lines.append(f"| {kind} {'highway' if highway else 'plain'} | {depth} | {activation} "
                     f"| {bias if highway else 'n/a'} | {group[0]['params']:,} | {group[0]['config']['lr']:g} "
                     f"| {loss:.2e} | {acc.max():.2f} ({acc.mean():.2f} ± {acc.std(ddof=1):.2f}) | {len(group)} |")
    Path("results/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_panels(runs, panels, metric, ylabel, path, scale=1, log=True, rows=1):
    """Small multiples of plain vs. highway: the median over seeds, shaded from the best seed to the worst."""
    fig, axes = plt.subplots(rows, len(panels) // rows, figsize=(2.6 * len(panels) / rows, 2.5 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (title, plain, highway) in zip(axes.flat, panels):
        for config, color, label in ((plain, PLAIN, "Plain"), (highway, HIGHWAY, "Highway")):
            if runs.get(key(config)):
                values = scale * np.array([run["history"][metric] for run in runs[key(config)]])
                epochs = range(1, values.shape[1] + 1)
                ax.plot(epochs, np.median(values, axis=0), color=color, label=label)
                ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=0.2, linewidth=0)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, yscale="log" if log else "linear")
        ax.label_outer()
    axes.flat[0].legend()
    fig.savefig(path)


def curves(runs, device):
    loss = ("train_loss", "Training cross-entropy")
    plot_panels(runs, [(f"{d} layers", mlp(d, False), mlp(d)) for d in (10, 20, 50, 100)], *loss, "figures/depth.png")
    plot_panels(runs, [(f"Gate bias {b:g}", mlp(50, False), mlp(50, gate_bias=b)) for b in BIASES], *loss,
                "figures/bias.png", rows=2)
    plot_panels(runs, [(f"{a}, 50 layers", mlp(50, False, a), mlp(50, True, a)) for a in ACTIVATIONS], *loss,
                "figures/activation.png")
    plot_panels(runs, [(f"{d} layers", cnn(d, False), cnn(d, True)) for d in (11, 19, 32)], "test_acc",
                "Test accuracy (%)", "figures/cnn.png", scale=100, log=False)


@torch.no_grad()
def capture(model, modules, images):
    """The outputs of `modules` for `images`, stacked as (module, image, unit)."""
    outputs = []
    hooks = [module.register_forward_hook(lambda _, __, output: outputs.append(output)) for module in modules]
    model(images)
    for hook in hooks:
        hook.remove()
    return torch.stack(outputs).cpu()


def heatmap(ax, matrix, colormap, title, **limits):
    ax.grid(False)
    ax.set(title=title, xlabel="Block", ylabel="Layer")
    ax.label_outer()
    return ax.imshow(matrix, aspect="auto", cmap=colormap, interpolation="nearest", **limits)


def gates(runs, device):
    """The paper's Figure 2 (gates and block outputs) and the follow-up's Figure 3 (differences between classes)."""
    images, labels = (tensor[:10000] for tensor in load_cifar10(device)["train"])
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharex=True, sharey=True)
    class_fig, class_axes = plt.subplots(2, len(CLASSES), figsize=(10, 5), sharex=True, sharey=True)
    for row, class_row, activation in zip(axes, class_axes, ACTIVATIONS):
        model, layers = best_model(runs, mlp(50, activation=activation), device)
        outputs = capture(model, [layer.gate for layer in layers], images)
        blocks = capture(model, layers, images[:1])[:, 0]
        biases = torch.stack([layer.gate[0].bias.detach().cpu() for layer in layers])
        limit = blocks.abs().max().item()
        for ax, (title, matrix, colormap, limits) in zip(row, (
                ("gate biases", biases, SEQUENTIAL, {}), ("mean gates", outputs.mean(1), SEQUENTIAL, dict(vmin=0)),
                ("gates, one image", outputs[:, 0], SEQUENTIAL, dict(vmin=0, vmax=1)),
                ("block outputs, one image", blocks, DIVERGING, dict(vmin=-limit, vmax=limit)))):
            fig.colorbar(heatmap(ax, matrix, colormap, f"{activation}: {title}", **limits), ax=ax)
        differences = [outputs[:, labels.cpu() == c].mean(1) - outputs.mean(1) for c in CLASSES]
        limit = max(difference.abs().max().item() for difference in differences)
        for ax, difference, name in zip(class_row, differences, CLASSES.values()):
            image = heatmap(ax, difference, DIVERGING, f"{activation}: {name}", vmin=-limit, vmax=limit)
        class_fig.colorbar(image, ax=class_row)
    fig.savefig("figures/gates.png")
    class_fig.savefig("figures/class_gates.png")


def gradients(runs, device):
    """Gradient size at every layer before any training, per image, averaged over five initializations."""
    images, labels = (tensor[:256] for tensor in load_cifar10(device)["train"])
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for config, color, label in ((mlp(50, False), PLAIN, "Plain"), (mlp(50), HIGHWAY, "Highway")):
        sizes = []
        for seed in range(5):
            torch.manual_seed(seed)
            x, outputs = images, []
            for module in build_model(config).to(device):
                x = module(x)
                if isinstance(module, (torch.nn.ReLU, Highway)):
                    x.retain_grad()
                    outputs.append(x)
            F.cross_entropy(x, labels).backward()
            sizes.append([output.grad.pow(2).mean().sqrt().item() * len(images) for output in outputs])
        ax.plot(range(1, 51), np.mean(sizes, axis=0), color=color, label=label)
    ax.set(xlabel="Layer", ylabel="Gradient size (RMS)", yscale="log")
    ax.legend()
    fig.savefig("figures/gradients.png")


def lesion(runs, device):
    """The follow-up's Figure 4: training loss when one layer's gates are forced shut, so it only copies its input."""
    images, labels = (tensor[:10000] for tensor in load_cifar10(device)["train"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, (title, config) in zip(axes, (("50-layer highway MLP", mlp(50)), ("19-layer highway CNN", cnn(19, True)))):
        model, layers = best_model(runs, config, device)
        losses = []
        for layer in layers:
            hook = layer.gate.register_forward_hook(lambda _, __, output: torch.zeros_like(output))
            losses.append(evaluate(model, images, labels)[0])
            hook.remove()
        ax.axhline(evaluate(model, images, labels)[0], color=MUTED, linewidth=1, label="No layer bypassed")
        ax.plot(range(1, len(losses) + 1), losses, color=HIGHWAY, marker="o", markersize=3, label="One layer bypassed")
        ax.set(title=title, xlabel="Bypassed highway layer", ylabel="Training cross-entropy", yscale="log")
        ax.legend()
    fig.savefig("figures/lesion.png")


def main():
    parts = dict(table=table, curves=curves, gates=gates, gradients=gradients, lesion=lesion)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="*", help=f"any of: {', '.join(parts)} (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    Path("figures").mkdir(exist_ok=True)
    for part in args.parts or parts:
        parts[part](load_runs(), args.device)


if __name__ == "__main__":
    main()
