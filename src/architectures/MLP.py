import torch.nn as nn

class MLP(nn.Module):
  '''
  Base Multi-layer Perceptron (MLP) architecture with option for random 
  sinusoidal (Fourier) features

  '''
  
  def __init__(self,
               input_dim,
               hidden_dims,
               output_dim,
               activation=nn.GELU(),
               fourier_features=None):
    '''
    --------
    Arguments
    --------
    
      input_dim: int
        The number of input dimensions.

      hidden_dims: list
        A list of the dimensions (number of neurons) for each hidden layer.

      output_dim: int
        The number of output dimensions.

      activation:
        Activation function for the hidden layers. Default is GELU activation.

      fourier_features: FourierFeatures
        An instance of the FourierFeatures class that engineers random 
        sinusoidal (Fourier) features for the inputs. Default is None, meaning
        no sinusoidal feature transformation of inputs.
    
    '''

    super().__init__()
    # An instance of random sinusoidal features class.
    self.fourier = fourier_features
    layers = []
    current_dim = input_dim
    if self.fourier is not None:
      current_dim = self.fourier.num_frequencies * 2

    # Define the MLP layers.
    for h in hidden_dims:
      linear_layer = nn.Linear(current_dim, h)
      nn.init.xavier_normal_(linear_layer.weight)
      layers.append(linear_layer)
      layers.append(activation)
      current_dim = h
    layers.append(nn.Linear(current_dim, output_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    '''
    Forward function for MLP with option to include random sinusoidal 
    features for the inputs.

    '''
    
    if self.fourier is not None:
      x = self.fourier(x)
    return self.net(x)