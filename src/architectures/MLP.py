import torch.nn as nn

class MLP(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dims,
               output_dim,
               activation=nn.GELU(),
               fourier_features=None):

    super().__init__()
    self.fourier = fourier_features
    layers = []
    current_dim = input_dim
    if self.fourier is not None:
      current_dim = self.fourier.num_frequencies * 2

    for h in hidden_dims:
      linear_layer = nn.Linear(current_dim, h)
      nn.init.xavier_normal_(linear_layer.weight)
      layers.append(linear_layer)
      layers.append(activation)
      current_dim = h
    layers.append(nn.Linear(current_dim, output_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    if self.fourier is not None:
      x = self.fourier(x)
    return self.net(x)