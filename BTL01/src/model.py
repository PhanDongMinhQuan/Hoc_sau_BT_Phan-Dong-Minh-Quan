import torch
from torch import nn


class GaussianMLP(nn.Module):
    """Two input coordinates, configurable hidden layers, one linear output."""
    def __init__(self, hidden):
        super().__init__()
        if not hidden or any(not isinstance(n, int) or n < 1 for n in hidden):
            raise ValueError("hidden must be a nonempty list of positive integers")
        layers = []
        width = 2
        for n in hidden:
            layers.extend([nn.Linear(width, n), nn.Tanh()])
            width = n
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def predict(model, x, device="cpu"):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
