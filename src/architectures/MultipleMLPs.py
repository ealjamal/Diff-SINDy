import torch
import torch.nn as nn

class MultipleMLPs(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_coeffs,
                 activation=nn.GELU(), fourier_features=None,
                 dropout_p=0.1):
        super().__init__()

        self.activation = activation
        self.dropout_p = dropout_p
        self.fourier = fourier_features

        if self.fourier is not None:
            encoded_dim = self.fourier.num_frequencies * 2
        else:
            encoded_dim = input_dim

        self.nets = nn.ModuleList([
            self._make_single_mlp(encoded_dim, hidden_dims)
            for _ in range(n_coeffs)
        ])

    def _make_single_mlp(self, input_dim, hidden_dims):
        layers = []
        current_dim = input_dim

        for h in hidden_dims:
            linear_layer = nn.Linear(current_dim, h)
            nn.init.xavier_normal_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout_p))
            current_dim = h

        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.fourier is not None:
            x = self.fourier(x)

        outputs = [net(x) for net in self.nets]
        return torch.cat(outputs, dim=-1)