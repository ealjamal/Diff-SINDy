import torch

def compute_derivatives(u_pred, 
                        inputs, 
                        lhs_deriv, 
                        deriv_library, 
                        input_names, 
                        add_ones):
    '''
    Compute derivatives from a given derivative library and the left-hand
    side (LHS) derivative of the PDE using automatic differentiation (AD). 
    Optionally, adds a column of ones for a PDE that involves a dforcing term.

    --------
    Arguments
    --------
    
      u_pred: torch.Tensor with requires_grad=True
        The approximate solution that is usually parameterized by an MLP.

      inputs: list of str
        The input names. For example ['x', 't'] denotes 1D space and time.

      lhs_derive: str
        A string denoting the LHS derivative. For example, 'u_t' denotes
        the first time derivative of u_pred and 'u_tt' the second time derivative.
        AD will be used to compute this derivative.

      deriv_library: list of str
        The list of derivatives that are possibly in the PDE. AD will be used to
        compute these derivatives.

      add_ones: bool
        If True, this will concatenate a columns of 1s in the 0th index
        to account for the forcing term in the PDE.
        
    --------
    Output
    --------
    
      lhs_d: torch.Tensor
        The LHS derivative of the approximated solution (u_pred). Same
        shape as u_pred.

      D: torch.Tensor
        The derivatives of the approximated solution (u_pred) specified in
        the derivative library, of shape (u_pred.shape[0], len(deriv_library) + 1)
        if add_ones=True because we add a column of ones, 
        (u_pred.shape[0], len(deriv_library)) otherwise.

    
    '''
    
    # Ensure the inputs require grad so we can take derivativs.
    inputs.requires_grad_(True)
    # helper dictionary denoting the order of inputs in the data.
    var_to_dim = {name: i for i, name in enumerate(input_names)}
    # A derivative cache so we do not take derivatives multiple times
    # when computing higher derivatives.
    deriv_cache = {'u': u_pred}

    # A helper function to compute a single derivative given a string.
    def _compute_derivative(term):
        # If the derivative is already computed return in.
        if term in deriv_cache:
            return deriv_cache[term]
        # If the term is just u, return the approximated solution.
        if term == 'u':
            deriv_cache['u'] = u_pred
            return u_pred

        # Find the derivative directions.
        suffix = term.split('_')[1]
        directions = list(suffix)
        d = u_pred
        # Compute the derivative.
        for var in directions:
          # The direction wrt which the derivatives are computed.
          dim = var_to_dim[var]
          # AD differentiation.
          d = torch.autograd.grad(d, inputs, torch.ones_like(d), 
                                  create_graph=True, retain_graph=True)[0][:, [dim]]
          deriv_cache[term] = d

        return d

    # Compute the LHS derivative.
    lhs_d = _compute_derivative(lhs_deriv)
    # Compute derivatives in the derivative library and add a one column if needed.
    if add_ones:
      D = torch.cat([torch.ones_like(lhs_d)] + [_compute_derivative(term) 
                                                for term in deriv_library], dim=1)
    else:
      D = torch.cat([_compute_derivative(term) 
                     for term in deriv_library], dim=1)

    return lhs_d, D