import torch

# The following functions will implement loss weighting schemes either with 
# or without the PDE loss included. We follow the treatment in 
# Wang et al. (2021) https://arxiv.org/abs/2001.04536.

def grad_norm(loss, 
              params, 
              eps=1e-8):
  '''
  Compute the L2 norm of the loss gradients wrt. the parameters.

  --------
  Arguments
  --------
    
    loss: float
      The loss for which we calculate the L2 norm of the gradient wrt. 
      the parameters.

    params:
      The parameters or weights that lead to the loss

    eps: float
      A small addition to the norm to prevent it from being zero and leading
      to divergences. Default is 1e-8.

    --------
    Output
    --------

    norm_grad: float
      The L2 norm of the loss gradient wrt. its parameters.

  '''
  
  # Collect the parameters that are learnable.
  params = [p for p in params if p.requires_grad]
  # If there are no parameters that are learnable we return 1.
  if len(params) == 0:
    return torch.tensor(1.0)
  # Take the gradient of the loss wrt. parameters.
  grads = torch.autograd.grad(loss, params,
                              retain_graph=True, create_graph=False)
  # Collect gradients with respect to parameters.
  grads = [g for g in grads if g is not None]

  # Calculate the L2 norm of gradients.
  norm_grad = torch.sqrt(sum(g.pow(2).sum() for g in grads)) + eps

  return norm_grad

def loss_weighting_data_wang(sol_model,
                             loss_ic,
                             loss_bc,
                             loss_data,
                             lambda_ic,
                             lambda_bc,
                             lambda_alpha):
    '''
    Perform moving average weighting on the initial condition (IC)
    and boundary condition (BC) losses with respect to the interior data loss.

    --------
    Arguments
    --------

      sol_model:
        The solution model given by some MLP architecture.

      loss_ic: 
        The loss of initial condition (IC).

      loss_bc:
        The loss of boundary conditions (BCs).

      loss_data:
        The loss of the interior data.

      lambda_ic: float
        The weight on the IC loss. This will be updated.

      lambda_bc: float
        The weight on the BC loss. This will be updated.

      lambda_alpha: float
        The weight of the update for lambda_ic and lambda_bc. The weight of the 
        previous value will be (1 - lambda_alpha). Therefore, lambda_alpha
        must be between 0 and 1.

    --------
    Output
    --------
    
      Return a tuple of updated lambda_ic and lambda_bc, using a moving average.
        
      lambda_ic: float
        The updated weight for the IC loss.

      lambda_bc: float
        The updated weight for the BC loss.

    '''

    # Calculate gradients for IC, BC, and interior data losses.
    g_ic = grad_norm(loss_ic, sol_model.parameters())
    g_bc = grad_norm(loss_bc, sol_model.parameters())
    g_data = grad_norm(loss_data, sol_model.parameters())

    # Take the max of the interior data loss gradient.
    g_ref = torch.max(g_data.detach())

    # Calculate the new weighting for IC and BC losses.
    lambda_ic_new = g_ref/g_ic.detach()
    lambda_bc_new = g_ref/g_bc.detach()

    # Perform moving average of the weights on IC and BC losses.
    lambda_ic = (1 - lambda_alpha) * lambda_ic + lambda_alpha * lambda_ic_new
    lambda_bc = (1 - lambda_alpha) * lambda_bc + lambda_alpha * lambda_bc_new

    return lambda_ic.detach(), lambda_bc.detach()

def loss_weighting_pde_wang(sol_model,
                            loss_ic,
                            loss_bc,
                            loss_data,
                            loss_pde,
                            lambda_ic,
                            lambda_bc,
                            lambda_data,
                            lambda_alpha):
    '''
    Perform moving average weighting on the initial condition (IC), 
    boundary condition (BC), and interior data losses with respect to 
    the PDE loss.

    --------
    Arguments
    --------

      sol_model:
        The solution model given by some MLP architecture.

      loss_ic: 
        The loss of initial condition (IC).

      loss_bc:
        The loss of boundary conditions (BCs).

      loss_data:
        The loss of the interior data.

      loss_pde:
        The loss of the PDE.

      lambda_ic: float
        The weight on the IC loss. This will be updated.

      lambda_bc: float
        The weight on the BC loss. This will be updated.

      lambda_data: float
        The weight on the interior data loss. This will be updated.

      lambda_alpha: float
        The weight of the update for lambda_ic and lambda_bc. The weight of the 
        previous value will be (1 - lambda_alpha). Therefore, lambda_alpha
        must be between 0 and 1.

    --------
    Output
    --------
    
      Return a tuple of updated lambda_ic, lambda_bc, lambda_data 
      using a moving average.
        
      lambda_ic: float
        The updated weight for the IC loss.

      lambda_bc: float
        The updated weight for the BC loss.

      lambda_data: float
        The updated weight for the interior data loss.

    '''

    # Calculate gradients for IC, BC, interior data, and PDE losses.
    g_ic = grad_norm(loss_ic, sol_model.parameters())
    g_bc = grad_norm(loss_bc, sol_model.parameters())
    g_data = grad_norm(loss_data, sol_model.parameters())
    g_pde = grad_norm(loss_pde, sol_model.parameters())

    # Take the max of the PDE loss gradient.
    g_ref = torch.max(g_pde.detach())

    # Calculate the new weighting for IC, BC, and interior data losses.
    lambda_ic_new = g_ref/g_ic.detach()
    lambda_bc_new = g_ref/g_bc.detach()
    lambda_data_new = g_ref/g_data.detach()

    # Perform moving average of the weights on IC, BC, and interior data losses.
    lambda_ic = (1 - lambda_alpha) * lambda_ic + lambda_alpha * lambda_ic_new
    lambda_bc = (1 - lambda_alpha) * lambda_bc + lambda_alpha * lambda_bc_new
    lambda_data = (1 - lambda_alpha) * lambda_data + lambda_alpha * lambda_data_new

    return lambda_ic.detach(), lambda_bc.detach(), lambda_data.detach()

def lambda_schedule(step, warmup_steps, max_lambda):
    '''
    A scheduler to gradually increase the weighting of the loss.

    --------
    Arguments
    --------
      step: int
        The current epoch.

      warm_steps: int
        The number of epochs for which to gradually increase the loss weight.

      max_lambda: float
        The maximum weighting that will be achieved. After warm_steps epochs,
        this will be the value of the weighting for all subsequent epochs.
    
    '''

    if step < warmup_steps:
        return max_lambda * (step / warmup_steps)
    
    return max_lambda