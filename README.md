# Diff-SINDy:
## A Differentiable Framework for Discovering Partial Differential Equations from Data

<p align="center">
  <img src="logo.png" width="300" title="logo">
</p>

## Dependencies

`torch` `PySR`.

## Quickstart

We provide more detailed examples in the `examples` directory with the data started data
in `examples/Data`. We provide a quickstart example below.

```
from architectures.FourierFeatures import FourierFeatures
from architectures.MLP import MLP
from architectures.MultiTaskMLP import MultiTaskMLP
from training.Trainer import Trainer
from inference.PDESR import PDESR
import scipy # only for data loading
```

### Training

```
data_file = "<path/to/data/data.mat>"
data = scipy.io.loadmat(data_file)

# Extract IC, BC, interior data with solutions at these points.
X_ic, U_ic, X_bc, U_bc, X_data, U_data = data

# Define collocation points
X_col = ...

# Define the solution network.
sol_fourier_features = FourierFeatures(input_dim=2,
                                       num_frequencies=20,
                                       scale=1.0)
sol_model = MLP(input_dim=2,
                hidden_dims=[128, 128],
                output_dim=1,
                activation=torch.nn.GELU(),
                fourier_features=sol_fourier_features)
sol_optimizer = torch.optim.Adam(sol_model.parameters(), lr=1e-4)

# Define coefficient network:
coeffs_model = MultiTaskMLP(input_dim=2
                            shared_hidden_dims=[256, 256, 256],
                            head_num_layers=1,
                            n_coeffs=5,
                            activation=torch.nn.GELU(),
                            fourier_features=None).to(device=DEVICE)
coeffs_optimizer = torch.optim.Adam(coeffs_model.parameters(), lr=1e-4)

# Define trainer for two-stage training procedure.
trainer = Trainer(lhs_deriv='u_t',
                  deriv_library=['u_x', 'u_xx', 'u_tt', 'u_tx'],
                  add_const_coeff=True,
                  input_names=['x', 't'],
                  coeffs_input_names=['x', 't', 'u'],
                  sol_optimizer=sol_optimizer,
                  coeffs_optimizer=coeffs_optimizer,
                  n_epochs=1000,
                  sol_warmup_n_epochs=0,
                  sol_update_per_epoch=5,
                  coeffs_update_per_epoch=5,
                  lambda_pde_warmup_epochs=500,
                  max_lambda_pde=1.0,
                  sol_warmup_loss_weighting=True,
                  fit_loss_weighting=True,
                  lambda_ic=1.0,
                  lambda_bc=1.0,
                  lambda_data=1.0,
                  lambda_pde=1.0,
                  lambda_alpha=0.9,
                  verbosity=100)

# Train the solution and coefficients models.
training_losses = trainer.fit(sol_model=sol_model,
                              coeffs_model=coeffs_model,
                              X_ic=xt_ic, 
                              U_ic=u_ic, 
                              X_bc=xt_bc, 
                              U_bc=u_bc, 
                              X_data=xt_data, 
                              U_data=u_data, 
                              X_col=xt_col)         
```

### Inference

```
# Define symbolic regression context with PySR by defining loss, number of iterations,
# conditions, and complexities.
pdesr = PDESR(trainer=trainer,
              niterations=100,
              early_stop_condition=(
                  "stop_if(loss, complexity) = loss < 1e-10 && complexity < 10"
                  ),
              timeout_in_seconds=5 * 60,
              maxsize=10,
              maxdepth=10,
              binary_operators=["*", "+", "-", "/"],
              unary_operators=["square", "cube", "exp", "cos", "sin"],
              constraints={
                  "/": (-1, 3),
                  "square": 3,
                  "cube": 3,
                  "exp": 3,
                  "sin": 3,
                  "cos": 3
                  },
              complexity_of_operators={"/": 2, "exp": 4, "cos": 2, "sin": 2},
              complexity_of_constants=1,
              model_selection='score',
              output_directory=None)

# Fit for the symbolic expressions of the coefficients and place results for
# each coefficient in a dictionary denoting the possible expressions with complexity,
# loss, and score.
coeffs_expressions = pdesr.fit(xt_data)      
```

## Website
Diff-SINDy is still in development and we are looking for calloborators. For contact information as well as 
a more detailed overview, please visit our website: https://ealjamal.github.io/Diff-SINDy/.