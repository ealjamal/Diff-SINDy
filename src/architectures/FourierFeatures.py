import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
  def __init__(self, input_dim, num_frequencies=20, scale=1.0):

    super().__init__()
    self.num_frequencies = num_frequencies
    self.B = nn.Parameter(torch.randn(input_dim, num_frequencies) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = 2 * torch.pi * x @ self.B
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)