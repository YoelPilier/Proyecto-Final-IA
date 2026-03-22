import torch
import torch.nn as nn
from datos import generos

n = len(generos)


class Modelo(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(n, 10)
        self.fc2 = nn.Linear(10, n)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


modelo = Modelo()