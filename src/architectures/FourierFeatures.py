import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
  '''
  Given some inputs, this is a module giving random Gaussian
  sinusoidal (Fourier) features.

  '''
  def __init__(self, input_dim, num_frequencies=20, scale=1.0):
    '''
    Args:
    
    input_dim: int
      The number of input dimensions.

    num_frequencies: int
      The number of random sinusoidal frequencies to sample from Gaussian. The 
      number of output sinusoidal features will be 2 * num_frequencies, since
      the features will have a cosine and a sine. Default is 20.

    scale: float
      The standard deviation of the Gaussian from which the frequencies
      are samples. Default is 1.0.

    '''

    super().__init__()
    self.num_frequencies = num_frequencies
    # The random frequency parameters. These are not learnable parameters.
    self.B = nn.Parameter(torch.randn(input_dim, num_frequencies) * scale, requires_grad=False)

  def forward(self, x):
    '''
    Forward function that results in 2 * num_frequencies sinusoidal features.
    
    '''
    x_proj = 2 * torch.pi * x @ self.B
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)