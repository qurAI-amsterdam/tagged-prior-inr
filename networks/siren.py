"""SIREN: Sinusoidal Representation Network for implicit neural registration."""

import numpy as np
import torch
from torch import nn


class Siren(nn.Module):
    """
    Dense neural network with sine activation functions (SIREN).

    Parameters
    ----------
    layers : list of int  node counts per layer, e.g. [3, 256, 256, 256, 3]
    weight_init : bool  use the SIREN weight initialisation scheme
    omega : float  frequency multiplier applied inside sin activations
    """

    def __init__(self, layers, weight_init=True, omega=30):
        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega

        layer_list = []
        for i in range(self.n_layers):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        layer_list[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        layer_list[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        """Apply sine activations to all hidden layers; linear output on the last."""
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))
        return self.layers[-1](x)
