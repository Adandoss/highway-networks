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
from experiments import CNN_DEPTHS, GATE_BIASES, MLP_DEPTHS, cnn, mlp
from models import ACTIVATIONS, Highway, build_model
from train import CHECKPOINTS, EVAL_IMAGES, RESULTS, evaluate

FIGURES = Path(__file__).parent / "figures"
ACTIVATION_NAMES = {"relu": "ReLU", "tanh": "tanh"}
CLASSES = {0: "airplane", 1: "automobile", 3: "cat", 8: "ship"}  # four of the ten, by label

# Plain networks are orange and highway networks blue in every figure; gray marks a reference.
PLAIN, HIGHWAY, MUTED = "#eb6834", "#2a78d6", "#898781"
INK, AXIS, GRID = "#52514e", "#c3c2b7", "#e1e0d9"
# Magnitudes: one blue ramp from light to dark, in even steps of lightness.
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
# Signed values: blue below zero, red above, a neutral gray at zero.
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
# One line per gate bias, light to dark. The ramp's lightest quarter would be too faint for a line on white.
BIAS_COLORS = dict(zip(GATE_BIASES, SEQUENTIAL(np.linspace(0.25, 1, len(GATE_BIASES)))))

plt.switch_backend("Agg")
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.4, "legend.frameon": False,
    "figure.constrained_layout.use": True, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def config_key(config):
    """What tells the experiments apart: a configuration without its learning rate, seed and number of epochs."""
    return config["kind"], config["depth"], config["highway"], config["activation"], config["gate_bias"]


def load_runs():
    """The final runs that did not diverge, grouped by configuration, with one result per seed."""
    runs = {}
    for path in sorted(RESULTS.glob("*.json")):
        run = json.loads(path.read_text())
        if not run["diverged"]:
            runs.setdefault(config_key(run["config"]), []).append(run)
    return runs


def load_model(run, device):
    """A trained network, rebuilt from the configuration saved next to its weights."""
    saved = torch.load(CHECKPOINTS / f"{run['name']}.pt", map_location=device, weights_only=True)
    model = build_model(saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return model


def best_model(runs, config, device):
    """The seed with the lowest training loss: the paper also inspects its best network (Section 4)."""
    return load_model(min(runs[config_key(config)], key=lambda run: run["history"]["train_loss"][-1]), device)


def highway_layers(model):
    return [module for module in model if isinstance(module, Highway)]


def log_scale(ax):
    """A log y-axis, ticked only at powers of ten."""
    ax.set_yscale("log")
    ax.minorticks_off()


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)


def table(runs, device):
    """Re-evaluate the saved weights and report them in the follow-up's format: best (mean ± std)."""
    test = load_cifar10(device)["test"]
    lines = ["| Network | Layers | Activation | Gate bias | Parameters | Learning rate | Training loss "
             "| Test accuracy (%) | Seeds finished |", "|---|---|---|---|---|---|---|---|---|"]
    for (kind, depth, highway, activation, gate_bias), group in sorted(runs.items()):
        accuracy = np.array([100 * evaluate(load_model(run, device), *test)[1] for run in group])
        loss = np.median([run["history"]["train_loss"][-1] for run in group])
        lines.append(f"| {kind} {'highway' if highway else 'plain'} | {depth} | {activation} "
                     f"| {gate_bias if highway else 'n/a'} | {group[0]['params']:,} | {group[0]['config']['lr']:g} "
                     f"| {loss:.2e} | {accuracy.max():.2f} ({accuracy.mean():.2f} ± {accuracy.std(ddof=1):.2f}) "
                     f"| {len(group)} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot_panels(runs, panels, name, metric="train_loss", ylabel="Training cross-entropy", scale=1, log=True, rows=1,
                ylim=None):
    """Small multiples, one panel per (title, plain configuration, highway configuration): the median over seeds,
    shaded from the best seed to the worst."""
    columns = len(panels) // rows
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.5 * rows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (title, plain, highway) in zip(axes.flat, panels):
        for config, color, label in ((plain, PLAIN, "Plain"), (highway, HIGHWAY, "Highway")):
            if config_key(config) in runs:
                values = scale * np.array([run["history"][metric] for run in runs[config_key(config)]])
                epochs = range(1, values.shape[1] + 1)
                ax.plot(epochs, np.median(values, axis=0), color=color, label=label)
                ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=0.2, linewidth=0)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        if log:
            log_scale(ax)
        if ylim:
            ax.set_ylim(ylim)
        ax.label_outer()
    axes.flat[0].legend()
    save(fig, name)


def curves(runs, device):
    """Per-epoch curves: the MLPs' training loss by depth, gate bias and activation, and the CNNs' test accuracy."""
    depth = [(f"{d} layers", mlp(d, highway=False), mlp(d)) for d in MLP_DEPTHS]
    plot_panels(runs, depth, "depth.png")
    bias = [(f"Gate bias {b:g}", mlp(50, highway=False), mlp(50, gate_bias=b)) for b in GATE_BIASES]
    plot_panels(runs, bias, "bias.png", rows=2)
    activation = [(f"{ACTIVATION_NAMES[a]}, {d} layers", mlp(d, highway=False, activation=a), mlp(d, activation=a))
                  for d in (50, 100) for a in ACTIVATIONS]
    plot_panels(runs, activation, "activation.png", rows=2)
    accuracy = [(f"{d} layers", cnn(d, highway=False), cnn(d)) for d in CNN_DEPTHS]
    plot_panels(runs, accuracy, "cnn.png", metric="test_acc", ylabel="Test accuracy (%)", scale=100, log=False,
                ylim=(70, 95))  # zoomed in, so that final differences of about one point show


@torch.no_grad()
def capture(model, modules, images):
    """The outputs of `modules` for `images`, stacked as (module, image, unit)."""
    outputs = []
    hooks = [module.register_forward_hook(lambda _, __, output: outputs.append(output)) for module in modules]
    model(images)
    for hook in hooks:
        hook.remove()
    return torch.stack(outputs).cpu()


def heatmap(ax, matrix, title, **style):
    """One cell per highway layer (row) and block (column), both counted from 1."""
    layers, blocks = matrix.shape
    ax.grid(False)
    ax.set(title=title, xlabel="Block", ylabel="Highway layer")
    ax.label_outer()
    return ax.imshow(matrix, aspect="auto", interpolation="nearest", extent=(0.5, blocks + 0.5, layers + 0.5, 0.5),
                     **style)


def color_scale(matrix):
    """The blue ramp for values that are never negative, otherwise blue-gray-red with gray at zero."""
    if matrix.min() >= 0:
        return dict(cmap=SEQUENTIAL, vmin=0)
    limit = matrix.abs().max().item()
    return dict(cmap=DIVERGING, vmin=-limit, vmax=limit)


def plot_gates(models, images):
    """The paper's Figure 2: per highway layer and block, the gate bias, the gate output averaged over the images,
    and the gate output and the layer's output for one image."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharex=True, sharey=True)
    for row, (activation, model) in zip(axes, models.items()):
        layers = highway_layers(model)
        biases = torch.stack([layer.gate[0].bias.detach().cpu() for layer in layers])
        gates = capture(model, [layer.gate for layer in layers], images)  # (layer, image, block)
        outputs = capture(model, layers, images[:1])[:, 0]  # (layer, block)
        panels = [("gate biases", biases, dict(cmap=SEQUENTIAL)),
                  ("mean gates", gates.mean(1), dict(cmap=SEQUENTIAL, vmin=0)),
                  ("gates, one image", gates[:, 0], dict(cmap=SEQUENTIAL, vmin=0, vmax=1)),
                  ("block outputs, one image", outputs, color_scale(outputs))]
        for ax, (title, matrix, style) in zip(row, panels):
            fig.colorbar(heatmap(ax, matrix, f"{ACTIVATION_NAMES[activation]}: {title}", **style), ax=ax)
    save(fig, "gates.png")


def plot_class_gates(models, images, labels):
    """The follow-up's Figure 3: the mean gate output for the images of one class minus the mean over all images."""
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(10, 5), sharex=True, sharey=True)
    for row, (activation, model) in zip(axes, models.items()):
        layers = highway_layers(model)
        gates = capture(model, [layer.gate for layer in layers], images)
        differences = [gates[:, labels == label].mean(1) - gates.mean(1) for label in CLASSES]
        limit = max(difference.abs().max().item() for difference in differences)
        for ax, difference, name in zip(row, differences, CLASSES.values()):
            im = heatmap(ax, difference, f"{ACTIVATION_NAMES[activation]}: {name}",
                         cmap=DIVERGING, vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=row)  # one scale for the whole row
    save(fig, "class_gates.png")


def gates(runs, device):
    """What the gates do in the best 50-layer ReLU and tanh highway MLPs, on 10,000 training images."""
    images, labels = (tensor[:EVAL_IMAGES] for tensor in load_cifar10(device)["train"])
    models = {activation: best_model(runs, mlp(50, activation=activation), device) for activation in ACTIVATIONS}
    plot_gates(models, images)
    plot_class_gates(models, images, labels.cpu())


def rms(tensor):
    return tensor.pow(2).mean().sqrt().item()


def measure_layers(config, seed, images, labels, device):
    """Every layer's output in a new, untrained network: the size of the gradient and of the signal for one image
    (RMS), and how alike different images look (mean cosine similarity: for unit vectors, 1 - distance² / 2)."""
    torch.manual_seed(seed)
    x, outputs = images, []
    for module in build_model(config).to(device):
        x = module(x)
        if isinstance(module, (torch.nn.ReLU, Highway)):  # a layer's output
            x.retain_grad()
            outputs.append(x)
    F.cross_entropy(x, labels).backward()
    measures = []
    for output in outputs:
        gradient = rms(output.grad) * len(images)  # the loss is a mean over the images
        similarity = 1 - F.pdist(F.normalize(output.detach())).pow(2).mean().item() / 2
        measures.append((gradient, rms(output), similarity))
    return measures


def signals(runs, device):
    """Each layer of a 50-layer ReLU network before any training, the median over five initializations."""
    images, labels = (tensor[:256] for tensor in load_cifar10(device)["train"])
    networks = [("Plain", mlp(50, highway=False), PLAIN)]
    networks += [(f"Gate bias {b:g}", mlp(50, gate_bias=b), BIAS_COLORS[b]) for b in GATE_BIASES]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3))
    for label, config, color in networks:
        median = np.median([measure_layers(config, seed, images, labels, device) for seed in range(5)], axis=0)
        gradient, signal, similarity = median.T  # one value per layer
        print(f"{label}: gradient {gradient[0]:.2g} at layer 1, {gradient[-1]:.2g} at layer 50; "
              f"at layer 50 signal {signal[-1]:.2g}, similarity {similarity[-1]:.3f}")
        for ax, values in zip(axes, (gradient, signal, similarity)):
            ax.plot(range(1, len(values) + 1), values, color=color, label=label)
    for ax, ylabel in zip(axes, ("Gradient size (RMS)", "Signal size (RMS)", "Similarity between images")):
        ax.set(xlabel="Layer", ylabel=ylabel)
    log_scale(axes[0])
    log_scale(axes[1])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center")
    save(fig, "signals.png")


def lesion(runs, device):
    """The follow-up's Figure 4: training loss when one layer's gates are forced shut, so it only copies its input."""
    images, labels = (tensor[:EVAL_IMAGES] for tensor in load_cifar10(device)["train"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, (title, config) in zip(axes, [("50-layer highway MLP", mlp(50)), ("19-layer highway CNN", cnn(19))]):
        model = best_model(runs, config, device)
        intact = evaluate(model, images, labels)[0]
        bypassed = []
        for layer in highway_layers(model):
            hook = layer.gate.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
            bypassed.append(evaluate(model, images, labels)[0])
            hook.remove()
        ax.axhline(intact, color=MUTED, linewidth=1, label="No layer bypassed")
        ax.plot(range(1, len(bypassed) + 1), bypassed, color=HIGHWAY, marker="o", markersize=3,
                label="One layer bypassed")
        ax.set(title=title, xlabel="Bypassed highway layer", ylabel="Training cross-entropy")
        log_scale(ax)
        ax.legend()
    save(fig, "lesion.png")


if __name__ == "__main__":
    parts = dict(table=table, curves=curves, gates=gates, signals=signals, lesion=lesion)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="*", choices=parts, help="default: all of them")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    runs = load_runs()
    for part in args.parts or parts:
        parts[part](runs, args.device)
