import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10),
        )

    def forward(self, x):
        return self.network(x)
