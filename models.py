"""Highway networks and their plain partners.

"The paper" is Srivastava, Greff, Schmidhuber: Highway Networks (arXiv 1505.00387).
"The follow-up" is the same authors' Training Very Deep Networks (arXiv 1507.06228)."""
from functools import partial

import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}

# MLP layer widths for equal parameters per layer (paper, Section 3.1). A highway layer holds two 50 x 50 layers,
# H and T: 2 * (50^2 + 50) = 5100 parameters. A plain layer holds one 71 x 71 layer: 71^2 + 71 = 5112.
HIGHWAY_WIDTH = 50
PLAIN_WIDTH = 71

# The output channels of every conv layer, stage by stage, with max pooling between the stages. 11 and 19 layers are
# FitNet 1 and FitNet 4 (Romero et al. 2014, Table 7). For 32 layers the follow-up (Section 3.3.1) only gives the
# alternating 3x3 and 1x1 kernels (alternate=True: every second layer is 1x1), so the channels are our own choice.
# A plain network has no gates to pay for, so its layers are plain_scale times wider, for about equal parameters.
CNN_LAYOUTS = {
    11: dict(stages=[[16, 16, 16], [32, 32, 32], [48, 48, 64]], alternate=False, plain_scale=1.18),
    19: dict(stages=[[32, 32, 32, 48, 48], [80] * 6, [128] * 6], alternate=False, plain_scale=1.37),
    32: dict(stages=[[32] * 10, [64] * 10, [96] * 10], alternate=True, plain_scale=1.36),
}


def he(layer):
    """He et al. (2015) initialization, used for all weights (paper, Section 3.1)."""
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)
    return layer


def linear(n_in, n_out):
    return he(nn.Linear(n_in, n_out))


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


def build_mlp(depth, highway, activation, gate_bias):
    """One plain layer, then depth - 1 plain or highway layers, then softmax (paper, Section 3.1)."""
    act = ACTIVATIONS[activation]
    width = HIGHWAY_WIDTH if highway else PLAIN_WIDTH
    layers = [nn.Flatten(), linear(3 * 32 * 32, width), act()]
    for _ in range(depth - 1):
        if highway:
            layers.append(Highway(partial(linear, width, width), act, gate_bias))
        else:
            layers += [linear(width, width), act()]
    return nn.Sequential(*layers, linear(width, 10))


def build_cnn(depth, highway, activation, gate_bias):
    """Stages of conv layers with 2x2 max pooling between them, then global max pooling, a 500-unit layer and softmax
    (FitNets, Appendix A.1.2)."""
    layout, act = CNN_LAYOUTS[depth], ACTIVATIONS[activation]
    scale = 1 if highway else layout["plain_scale"]
    stages = [[round(c * scale) for c in stage] for stage in layout["stages"]]
    layers, channels = [], 3
    for s, stage in enumerate(stages):
        for i, out in enumerate(stage):
            kernel = 1 if layout["alternate"] and i % 2 else 3
            if highway and out == channels:
                layers.append(Highway(partial(conv, out, out, kernel), act, gate_bias))
            else:  # Eq. 3 needs equal shapes, so a layer that changes the channel count is plain (paper, Section 2.1)
                layers += [conv(channels, out, kernel), act()]
                channels = out
        last = s == len(stages) - 1
        layers.append(nn.AdaptiveMaxPool2d(1) if last else nn.MaxPool2d(2))
    return nn.Sequential(*layers, nn.Flatten(), linear(channels, 500), act(), linear(500, 10))


def build_model(config):
    """The network a configuration from experiments.py describes."""
    build = {"mlp": build_mlp, "cnn": build_cnn}[config["kind"]]
    return build(config["depth"], config["highway"], config["activation"], config["gate_bias"])
