import torch
import torch.nn as nn

class MultiTaskMLP(nn.Module):
    '''
    A multi-task MLP with a shared backbone and a separate head for 
    each coefficient.

    '''

    def __init__(self, 
                 input_dim,
                 shared_hidden_dims,
                 head_num_layers,
                 n_coeffs,
                 activation=nn.GELU(),
                 fourier_features=None):
        '''
        --------
        Arguments
        --------
        
            input_dim: int
                The number of input dimensions.

            shared_hidden_dims: list
                A list of the dimensions (number of neurons) for each hidden layer
                of the sahred backbone.

            head_num_layers: int
                The number of layers of the heads.

            n_coeffs: int
                The number of separate MLPs to design. This will be the number
                of coefficients.

            activation:
                Activation function for the hidden layers. Default is GELU activation.

            fourier_features: FourierFeatures
                An instance of the FourierFeatures class that engineers random 
                sinusoidal (Fourier) features for the inputs. Default is None, meaning
                no sinusoidal feature transformation of inputs.
        
        '''
        
        super().__init__()
        self.fourier = fourier_features
        self.head_num_layers = head_num_layers
        self.activation = activation
        current_dim = input_dim
        if self.fourier is not None:
            current_dim = fourier_features.num_frequencies * 2

        # Design the shared backbone.
        layers = []
        for h in shared_hidden_dims:
            linear_layer = nn.Linear(current_dim, h)
            nn.init.xavier_normal_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(activation)
            current_dim = h
        self.shared_net = nn.Sequential(*layers)

        # Design the heads attached to the shared backbone.
        self.heads = nn.ModuleList([self._make_head(current_dim)
                                    for _ in range(n_coeffs)])

    def _make_head(self, dim):
        '''
        Helper function to design a single head.

        --------
        Arguments
        --------

        dims: list
            Input dimension of the head.

        --------
        Output
        --------

        Instance of torch.nn.Sequential for the design of the head.
        
        '''

        layers = []
        for _ in range(self.head_num_layers - 1):
            linear_layer = nn.Linear(dim, dim)
            nn.init.xavier_normal_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(self.activation)
        layers.append(nn.Linear(dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward function outputting the results of each separate head.

        '''

        if self.fourier is not None:
            x = self.fourier(x)
        h = self.shared_net(x)
        coeffs = [head(h) for head in self.heads]
        return torch.cat(coeffs, dim=-1)