"""Highway networks and their plain partners.

"The paper" is Srivastava, Greff, Schmidhuber: Highway Networks (arXiv 1505.00387).
"The follow-up" is the same authors' Training Very Deep Networks (arXiv 1507.06228)."""
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}

# Equal parameters per layer (paper, Section 3.1): 2 * (50^2 + 50) = 5100 and 71^2 + 71 = 5112.
MLP_WIDTH = {True: 50, False: 71}

# Channels of every conv layer, block by block. 11 and 19 layers are FitNet 1 and FitNet 4 (Romero et al. 2014,
# Table 7). For 32 layers the follow-up (Section 3.3.1) only gives the alternating 3x3 and 1x1 kernels, so the
# channels are our own choice. gate_bias follows the follow-up (Section 2.2), plain_scale evens out the plain partner.
CNN_LAYOUTS = {
    11: dict(blocks=[[16, 16, 16], [32, 32, 32], [48, 48, 64]], alternate=False, gate_bias=-1.0, plain_scale=1.18),
    19: dict(blocks=[[32, 32, 32, 48, 48], [80] * 6, [128] * 6], alternate=False, gate_bias=-2.0, plain_scale=1.37),
    32: dict(blocks=[[32] * 10, [64] * 10, [96] * 10], alternate=True, gate_bias=-3.0, plain_scale=1.36),
}


def he(layer):
    """He et al. (2015) initialization, used for all weights (paper, Section 3.1)."""
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)
    return layer


def conv(c_in, c_out, kernel):
    return he(nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2))  # zero padding keeps the size


class Highway(nn.Module):
    """One highway layer: y = H(x) * T(x) + x * (1 - T(x)), the paper's Eq. 3.

    H is `transform` and T is `gate`. T near 1 makes this a plain layer, T near 0 copies x to the output.
    Eq. 3 only multiplies and adds element by element, so `make_layer` can build linear or conv layers alike,
    as long as they keep the shape of their input."""

    def __init__(self, make_layer, activation, gate_bias):
        super().__init__()
        self.transform = nn.Sequential(make_layer(), activation())
        self.gate = nn.Sequential(make_layer(), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, gate_bias)  # negative: starts nearly closed (paper, Section 2.2)

    def forward(self, x):
        t = self.gate(x)
        return self.transform(x) * t + x * (1 - t)


def mlp(depth, highway, activation, gate_bias):
    """One plain layer, then depth - 1 plain or highway layers, then softmax (paper, Section 3.1)."""
    act, width = ACTIVATIONS[activation], MLP_WIDTH[highway]
    make_layer = lambda: he(nn.Linear(width, width))  # noqa: E731
    layers = [nn.Flatten(), he(nn.Linear(3 * 32 * 32, width)), act()]
    for _ in range(depth - 1):
        layers += [Highway(make_layer, act, gate_bias)] if highway else [make_layer(), act()]
    return nn.Sequential(*layers, he(nn.Linear(width, 10)))


def cnn(depth, highway, activation, gate_bias):
    """Conv blocks with max pooling between them, global pooling, a 500-unit layer and softmax (FitNets, A.1.2)."""
    layout, act = CNN_LAYOUTS[depth], ACTIVATIONS[activation]
    layers, channels = [], 3
    for b, block in enumerate(layout["blocks"]):
        for i, out in enumerate(block):
            kernel = 1 if layout["alternate"] and i % 2 else 3
            if highway and out == channels:
                layers.append(Highway(lambda: conv(out, out, kernel), act, gate_bias))
            else:  # Eq. 3 needs equal shapes, so a plain layer changes the channel count (paper, Section 2.1)
                out = out if highway else round(out * layout["plain_scale"])
                layers += [conv(channels, out, kernel), act()]
                channels = out
        layers.append(nn.MaxPool2d(2) if b < len(layout["blocks"]) - 1 else nn.AdaptiveMaxPool2d(1))
    return nn.Sequential(*layers, nn.Flatten(), he(nn.Linear(channels, 500)), act(), he(nn.Linear(500, 10)))


def build_model(config):
    builder = {"mlp": mlp, "cnn": cnn}[config["kind"]]
    return builder(config["depth"], config["highway"], config["activation"], config["gate_bias"])
