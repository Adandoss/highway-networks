"""One training run: SGD with momentum, metrics every epoch, safe checkpoints and resume."""
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from data import batches, load_cifar10
from models import build_model

RESULTS = Path(__file__).parent / "results"
CHECKPOINTS = Path(__file__).parent / "checkpoints"
BATCH_SIZE = 128
MOMENTUM = 0.9
MAX_GRAD_NORM = 10.0  # gradient clipping: one exploding batch must not end a run
CHECKPOINT_EVERY = 10  # epochs
EVAL_IMAGES = 10_000  # the metrics after each epoch use the first 10,000 images of every split


def run_name(config):
    """A name that identifies the run, so experiments can share runs and a restart can find them."""
    variant = f"highway-b{config['gate_bias']:g}" if config["highway"] else "plain"
    return f"{config['kind']}-{config['depth']}-{variant}-{config['activation']}-lr{config['lr']:g}-s{config['seed']}"


@torch.no_grad()
def evaluate(model, images, labels):
    """Mean cross-entropy and accuracy over a whole dataset."""
    logits = torch.cat([model(batch) for batch in images.split(1000)])
    return F.cross_entropy(logits, labels).item(), (logits.argmax(1) == labels).sum().item() / len(labels)


def save_atomic(state, path):
    """Write to a temporary file first, so a crash can never leave a broken checkpoint behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def make_scheduler(optimizer, config):
    """Exponential decay for the fully connected networks, fixed steps for the rest (follow-up, Section 3)."""
    if config["kind"] == "mlp":
        return ExponentialLR(optimizer, gamma=0.98)
    epochs = config["epochs"]
    return MultiStepLR(optimizer, milestones=[epochs // 2, 3 * epochs // 4], gamma=0.1)


def train_epoch(model, optimizer, images, labels, augment):
    """One pass over the training set in shuffled mini-batches."""
    for batch_images, batch_labels in batches(images, labels, BATCH_SIZE, augment):
        optimizer.zero_grad()
        F.cross_entropy(model(batch_images), batch_labels).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()


def train_run(config, device="cpu"):
    """Train one network and return its results. Finished runs are read from disk, unfinished ones resume.
    A search run (val_size > 0) keeps only its metrics. A final run also keeps its weights and config,
    which is all it takes to rebuild the network on another machine."""
    name = run_name(config)
    folder = "search" if config["val_size"] else ""
    result_path = RESULTS / folder / f"{name}.json"
    checkpoint_path = CHECKPOINTS / folder / f"{name}.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["config"] != config:  # the name leaves things out, such as the number of epochs
            raise ValueError(f"{result_path} was trained with a different configuration")
        return result

    torch.manual_seed(config["seed"])
    data = load_cifar10(device, config["val_size"])
    model = build_model(config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), config["lr"], MOMENTUM, weight_decay=config["weight_decay"])
    scheduler = make_scheduler(optimizer, config)
    history, start = {}, 0
    if checkpoint_path.exists():  # an unfinished run: continue where it stopped
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["rng"].cpu())
        history, start = saved["history"], saved["epoch"]

    diverged = False
    for epoch in range(start, config["epochs"]):
        began = time.perf_counter()
        train_epoch(model, optimizer, *data["train"], config["augment"])
        scheduler.step()
        for split, (images, labels) in data.items():
            loss, accuracy = evaluate(model, images[:EVAL_IMAGES], labels[:EVAL_IMAGES])
            history.setdefault(f"{split}_loss", []).append(loss)
            history.setdefault(f"{split}_acc", []).append(accuracy)
        history.setdefault("seconds", []).append(time.perf_counter() - began)
        diverged = not math.isfinite(history["train_loss"][-1])
        if diverged:
            break
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_atomic(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), rng=torch.get_rng_state(),
                             history=history, epoch=epoch + 1), checkpoint_path)
            print(f"{name}: epoch {epoch + 1}/{config['epochs']}, train loss {history['train_loss'][-1]:.4f}, "
                  f"{history['seconds'][-1]:.1f} s/epoch", flush=True)

    if config["val_size"] or diverged:
        checkpoint_path.unlink(missing_ok=True)
    else:
        save_atomic(dict(config=config, state_dict=model.state_dict()), checkpoint_path)
    result = dict(name=name, config=config, diverged=diverged, history=history,
                  params=sum(p.numel() for p in model.parameters()))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result))  # written last: this file marks the run as finished
    print(f"{name}: {'diverged' if diverged else 'finished'}", flush=True)
    return result
