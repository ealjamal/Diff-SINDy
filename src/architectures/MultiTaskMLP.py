import torch
import torch.nn as nn

class MultiTaskMLP(nn.Module):
    def __init__(self, 
                 input_dim,
                 shared_hidden_dims,
                 head_num_layers,
                 n_coeffs,
                 activation=nn.GELU(),
                 fourier_features=None):
        
        super().__init__()
        self.fourier = fourier_features
        self.head_num_layers = head_num_layers
        self.activation = activation
        current_dim = input_dim
        if self.fourier is not None:
            current_dim = fourier_features.num_frequencies * 2

        layers = []
        for h in shared_hidden_dims:
            linear_layer = nn.Linear(current_dim, h)
            nn.init.xavier_normal_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(activation)
            current_dim = h
        self.shared_net = nn.Sequential(*layers)

        self.heads = nn.ModuleList([self._make_head(current_dim)
                                    for _ in range(n_coeffs)])

    def _make_head(self, dim):
        layers = []
        for _ in range(self.head_num_layers - 1):
            linear_layer = nn.Linear(dim, dim)
            nn.init.xavier_normal_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(self.activation)
        layers.append(nn.Linear(dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.fourier is not None:
            x = self.fourier(x)
        h = self.shared_net(x)
        coeffs = [head(h) for head in self.heads]
        return torch.cat(coeffs, dim=-1)