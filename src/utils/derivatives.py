import torch

def compute_derivatives(u_pred, 
                        inputs, 
                        lhs_deriv, 
                        deriv_library, 
                        input_names, 
                        add_ones):
    
    inputs.requires_grad_(True)
    var_to_dim = {name: i for i, name in enumerate(input_names)}
    deriv_cache = {'u': u_pred}

    def _compute_derivative(term):
        if term in deriv_cache:
            return deriv_cache[term]
        if term == 'u':
            deriv_cache['u'] = u_pred
            return u_pred

        suffix = term.split('_')[1]
        directions = list(suffix)
        d = u_pred
        for var in directions:
          dim = var_to_dim[var]
          d = torch.autograd.grad(d, inputs, torch.ones_like(d), create_graph=True, retain_graph=True)[0][:, [dim]]
          deriv_cache[term] = d

        return d

    lhs_d = _compute_derivative(lhs_deriv)
    if add_ones:
      D = torch.cat([torch.ones_like(lhs_d)] + [_compute_derivative(term) for term in deriv_library], dim=1)
    else:
      D = torch.cat([_compute_derivative(term) for term in deriv_library], dim=1)

    return lhs_d, D