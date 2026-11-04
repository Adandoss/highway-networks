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

BATCH_SIZE, MOMENTUM, CHECKPOINT_EVERY = 128, 0.9, 10


def run_name(config):
    """A name that identifies the run, so experiments can share runs and a restart can find them."""
    kind = f"highway-b{config['gate_bias']:g}" if config["highway"] else "plain"
    return f"{config['kind']}-{config['depth']}-{kind}-{config['activation']}-lr{config['lr']:g}-s{config['seed']}"


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


def train_run(config, device="cpu"):
    """Train one network and return its results. Finished runs are read from disk, unfinished ones resume.
    A search run (val_size > 0) keeps only its metrics. A final run also keeps its weights and config,
    which is all it takes to rebuild the network on another machine."""
    subfolder, name = "search" if config["val_size"] else "", run_name(config)
    result_path = Path("results") / subfolder / f"{name}.json"
    checkpoint = Path("checkpoints") / subfolder / f"{name}.pt"
    if result_path.exists():
        return json.loads(result_path.read_text())

    torch.manual_seed(config["seed"])
    data = load_cifar10(device, config["val_size"])
    model = build_model(config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), config["lr"], MOMENTUM, weight_decay=config["weight_decay"])
    # Exponential decay for the fully connected networks, fixed steps for the rest (follow-up, Section 3).
    epochs = config["epochs"]
    scheduler = (ExponentialLR(optimizer, 0.98) if config["kind"] == "mlp"
                 else MultiStepLR(optimizer, [epochs // 2, 3 * epochs // 4], 0.1))
    history, start, diverged = {}, 0, False
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, start = saved["history"], saved["epoch"]
        torch.set_rng_state(saved["rng"].cpu())

    for epoch in range(start, epochs):
        began = time.perf_counter()
        for images, labels in batches(*data["train"], BATCH_SIZE, config["augment"]):
            optimizer.zero_grad()
            F.cross_entropy(model(images), labels).backward()
            optimizer.step()
        scheduler.step()
        for split, (images, labels) in data.items():  # 10,000 training images are enough to follow the loss
            for metric, value in zip(("loss", "acc"), evaluate(model, images[:10000], labels[:10000])):
                history.setdefault(f"{split}_{metric}", []).append(value)
        history.setdefault("seconds", []).append(time.perf_counter() - began)
        if diverged := not math.isfinite(history["train_loss"][-1]):
            break
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_atomic(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                             scheduler=scheduler.state_dict(), epoch=epoch + 1, rng=torch.get_rng_state()), checkpoint)
            print(f"{name}: epoch {epoch + 1}/{epochs}, train loss {history['train_loss'][-1]:.4f}, "
                  f"{history['seconds'][-1]:.1f} s/epoch", flush=True)

    result = dict(name=name, config=config, diverged=diverged, history=history,
                  params=sum(p.numel() for p in model.parameters()))
    if config["val_size"] or diverged:
        checkpoint.unlink(missing_ok=True)
    else:
        save_atomic(dict(config=config, state_dict=model.state_dict()), checkpoint)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result))  # written last: this file marks the run as finished
    print(f"{name}: {'diverged' if diverged else 'finished'}", flush=True)
    return result
