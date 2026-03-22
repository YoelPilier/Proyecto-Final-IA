import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from datos import generos
from pytorch_model import modelo

criterio = nn.MSELoss()
opt = optim.Adam(modelo.parameters(), lr=0.01)


def texto_a_vector(texto):

    v = []

    for g in generos:
        v.append(1 if g in texto else 0)

    return np.array(v, dtype=np.float32)


def entrenar(texto):

    x = torch.tensor(texto_a_vector(texto)).float()

    y = x

    salida = modelo(x)

    loss = criterio(salida, y)

    opt.zero_grad()
    loss.backward()
    opt.step()