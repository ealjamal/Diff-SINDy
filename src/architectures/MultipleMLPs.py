import torch
import torch.nn as nn

class MultipleMLPs(nn.Module):
    '''
    A class to create multiple MLPs, one for each coefficient.
    
    '''

    def __init__(self, input_dim, hidden_dims, n_coeffs,
                 activation=nn.GELU(), fourier_features=None,
                 dropout_p=0.1):
        
        '''
        --------
        Arguments
        --------
        
        input_dim: int
            The number of input dimensions.

        hidden_dims: list
            A list of the dimensions (number of neurons) for each hidden layer
            for all MLPs.

        n_coeffs: int
            The number of separate MLPs to design. This will be the number
            of coefficients.

        activation:
            Activation function for the hidden layers. Default is GELU activation.

        fourier_features: FourierFeatures
            An instance of the FourierFeatures class that engineers random 
            sinusoidal (Fourier) features for the inputs. Default is None, meaning
            no sinusoidal feature transformation of inputs.

        dropout_p:
            Dropout probability during training. Default is 0.1.

        '''

        super().__init__()

        self.activation = activation
        self.dropout_p = dropout_p
        self.fourier = fourier_features

        # Define random sinusoidal (Fourier) features
        if self.fourier is not None:
            encoded_dim = self.fourier.num_frequencies * 2
        else:
            encoded_dim = input_dim

        # Make the separate MLPs.
        self.nets = nn.ModuleList([
            self._make_single_mlp(encoded_dim, hidden_dims)
            for _ in range(n_coeffs)
        ])

    def _make_single_mlp(self, input_dim, hidden_dims):
        '''
        Helper function to design a single MLP.

        --------
        Arguments
        --------
        
        input_dim: int
            The number of input dimensions.

        hidden_dims: list
            A list of the dimensions (number of neurons) for each
            hidden layer of the MLP. 

        --------
        Output
        --------

        Instance of torch.nn.Sequential for the design of the MLP.

        '''

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
        '''
        Forward function outputting the results of each separate MLP.

        '''
        
        if self.fourier is not None:
            x = self.fourier(x)

        outputs = [net(x) for net in self.nets]
        return torch.cat(outputs, dim=-1)