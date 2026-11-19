"""Matched architectures extracted from Ethan Vu's original notebook.

Default topology, surrogate, hard reset and normalization are preserved.
Constructor parameters replace notebook-global configuration.
"""
import torch
from torch import nn


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distance):
        ctx.save_for_backward(distance)
        return (distance >= 0).to(distance.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (distance,) = ctx.saved_tensors
        return grad_output * (1 - distance.abs()).clamp(min=0)


class LIFNeuron(nn.Module):
    def __init__(self, decay=0.5, threshold=1.0):
        super().__init__()
        if not 0 <= decay <= 1 or threshold <= 0:
            raise ValueError("decay must be in [0, 1] and threshold must be positive")
        self.decay, self.threshold = decay, threshold

    def forward(self, inputs):
        membrane = torch.zeros_like(inputs[:, 0])
        spikes = []
        for current in inputs.unbind(1):
            membrane = self.decay * membrane + current
            spike = SpikeFunction.apply(membrane - self.threshold)
            membrane = membrane * (1 - spike)
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class EveryTimestep(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        batch, time = x.shape[:2]
        out = self.layer(x.flatten(0, 1))
        return out.reshape(batch, time, *out.shape[1:])


class TEBN(nn.Module):
    """Batch/time/spatial statistics, followed by p[t] * BN(x[t])."""
    def __init__(self, channels, timesteps=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.timestep_scale = nn.Parameter(torch.ones(1, timesteps, 1, 1, 1))

    def forward(self, x):
        if x.shape[1] != self.timestep_scale.shape[1]:
            raise ValueError("input time dimension differs from configured timesteps")
        return self.bn(x.flatten(0, 1)).reshape_as(x) * self.timestep_scale


class PlainBN(nn.Module):
    """Shared batch/time statistics without the learnable temporal scales."""
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(x.flatten(0, 1)).reshape_as(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 3
        for channels in (64, 128, 256):
            for _ in range(2):
                layers.append(nn.Sequential(
                    nn.Conv2d(previous, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels), nn.ReLU()))
                previous = channels
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 256),
                                        nn.ReLU(), nn.Linear(256, 10))

    def forward(self, images):
        return self.classifier(self.features(images))


class SNN(nn.Module):
    def __init__(self, timesteps=4, use_tebn=True, decay=0.5, threshold=1.0):
        super().__init__()
        if not isinstance(timesteps, int) or timesteps < 1:
            raise ValueError("timesteps must be a positive integer")
        self.timesteps = timesteps
        layers, previous = [], 3
        for channels in (64, 128, 256):
            for _ in range(2):
                normalization = TEBN(channels, timesteps) if use_tebn else PlainBN(channels)
                layers.append(nn.Sequential(
                    EveryTimestep(nn.Conv2d(previous, channels, 3, padding=1, bias=False)),
                    normalization, LIFNeuron(decay, threshold)))
                previous = channels
            layers.append(EveryTimestep(nn.MaxPool2d(2)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            EveryTimestep(nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 256))),
            LIFNeuron(decay, threshold), EveryTimestep(nn.Linear(256, 10)))

    def forward(self, images):
        x = images.unsqueeze(1).expand(-1, self.timesteps, -1, -1, -1)
        return self.classifier(self.features(x)).mean(dim=1)


def make_model(name, timesteps=4):
    if name == "cnn":
        return CNN()
    if name in ("snn", "snn_plain"):
        return SNN(timesteps, use_tebn=name == "snn")
    raise ValueError(f"unknown model: {name}")
