import torch

def grad_norm(loss, 
              params, 
              eps=1e-8):
  
  params = [p for p in params if p.requires_grad]
  if len(params) == 0:
    return torch.tensor(1.0)
  grads = torch.autograd.grad(loss, params,
                              retain_graph=True, create_graph=False)
  grads = [g for g in grads if g is not None]

  return torch.sqrt(sum(g.pow(2).sum() for g in grads)) + eps

def lambda_weighting_data_wang(sol_model,
                               loss_ic,
                               loss_bc,
                               loss_data,
                               lambda_ic,
                               lambda_bc,
                               lambda_alpha):

    g_ic = grad_norm(loss_ic, sol_model.parameters())
    g_bc = grad_norm(loss_bc, sol_model.parameters())
    g_data = grad_norm(loss_data, sol_model.parameters())

    g_ref = torch.max(g_data.detach())

    lambda_ic_new = g_ref/g_ic.detach()
    lambda_bc_new = g_ref/g_bc.detach()

    lambda_ic = (1 - lambda_alpha) * lambda_ic + lambda_alpha * lambda_ic_new
    lambda_bc = (1 - lambda_alpha) * lambda_bc + lambda_alpha * lambda_bc_new

    return lambda_ic.detach(), lambda_bc.detach()

def lambda_weighting_pde_wang(sol_model,
                              loss_ic,
                              loss_bc,
                              loss_data,
                              loss_pde,
                              lambda_ic,
                              lambda_bc,
                              lambda_data,
                              lambda_alpha):

    g_ic = grad_norm(loss_ic, sol_model.parameters())
    g_bc = grad_norm(loss_bc, sol_model.parameters())
    g_data = grad_norm(loss_data, sol_model.parameters())
    g_pde = grad_norm(loss_pde, sol_model.parameters())

    g_ref = torch.max(g_pde.detach())

    lambda_ic_new = g_ref/g_ic.detach()
    lambda_bc_new = g_ref/g_bc.detach()
    lambda_data_new = g_ref/g_data.detach()

    lambda_ic = (1 - lambda_alpha) * lambda_ic + lambda_alpha * lambda_ic_new
    lambda_bc = (1 - lambda_alpha) * lambda_bc + lambda_alpha * lambda_bc_new
    lambda_data = (1 - lambda_alpha) * lambda_data + lambda_alpha * lambda_data_new

    return lambda_ic.detach(), lambda_bc.detach(), lambda_data.detach()

def lambda_schedule(step, warmup_steps, max_lambda):
    if step < warmup_steps:
        return max_lambda * (step / warmup_steps)
    
    return max_lambda