from utils.loss_weighting import *
from utils.derivatives import compute_derivatives
import torch.nn.functional as F

class Trainer:
    '''
    A class to implement the two stage training of Diff-SINDy PDE discovery
    with optional warmup and gradient balancing.
    
    '''

    def __init__(self, 
                 lhs_deriv,
                 deriv_library,
                 add_const_coeff,
                 input_names,
                 coeffs_input_names,
                 sol_optimizer,
                 coeffs_optimizer,
                 sol_lr_scheduler=None,
                 fit_lr_scheduler=None,
                 n_epochs=2000,
                 sol_warmup_n_epochs=0,
                 sol_update_per_epoch=10,
                 coeffs_update_per_epoch=10,
                 lambda_pde_warmup_epochs=1000,
                 max_lambda_pde=1.0,
                 sol_warmup_loss_weighting=True,
                 fit_loss_weighting=True,
                 lambda_ic=1.0,
                 lambda_bc=1.0,
                 lambda_data=1.0,
                 lambda_pde=1.0,
                 lambda_alpha=0.9,
                 verbosity=0):
        
        '''
        --------
        Arguments
        --------
        
            lhs_derive: str
                A string denoting the LHS derivative. For example, 'u_t' denotes
                the first time derivative of u_pred and 'u_tt' the second time derivative.
                AD will be used to compute this derivative.
                
            deriv_library: list of str
                The list of derivatives that are possibly in the PDE. AD will be used to
                compute these derivatives.
                
            add_const_coeff: bool
                If True, this will concatenate a columns of 1s in the 0th index
                to account for the forcing term in the PDE.
                
            input_names: list of str
                The input names. For example ['x', 't'] denotes 1D space and time.
                These must align with the dimensions of the input data.

            coeffs_input_names: list of str
                The input names of the variables that the coefficient will be a
                a function of. These are the inputs to the coefficients network(s).
                For example, ['x', 't'] or ['x', 't', 'u'], the latter includes
                the approximated solution.

            sol_optimizer: torch.optim
                The optimizer for the solution network.

            coeffs_optimizer: torch.optim
                The optimizer for the coefficients network.

            sol_lr_scheduler:
                Learning rate scheduler for the solution warm up only monitoring
                the warm up loss of IC, BC and interior data. Default is None, 
                meaning no learning rate scheduler.

            fit_lr_scheduler:
                Learning rate scheduler for the two-stage training procedure
                of the solution and coefficients network(s). This monitors the
                total loss of IC, BC, interior data, and PDE losses. Default is 
                None, meaning no learning rate scheduler.

            n_epochs: int
                Number of epochs to train. Default is 2000.

            sol_warmup_n_epochs: int
                Number of epochs to warm up the solution network (without PDE
                loss). Default is 0, meaning no warm up.

            coeffs_update_per_epoch: int
                The number of update steps for the solutions and coefficients 
                network(s) in each epoch. For each epoch in n_epochs, this is 
                how many times stage 1 (solution training) and stage 2 
                (coefficients training) will be optimized separately. Defualt
                is 10.

            lambda_pde_warmup_epochs: int
                The number of epochs for which lambda_pde will gradually increase
                to its maximum value. For subsequent epochs, the value of lambda_pde
                will be constant and equal to its max_lambda_pde value. Default
                is 1000, half of the default value for n_epochs.

            max_lambda_pde: float
                The maximum weighting on the PDE loss. Default is 1.

            sol_warmup_loss_weighting: bool
                If True, gradient balancing will be implemented while
                warming the solution network of the IC and BC losses
                wrt. the interior data loss. Default is True.

            fit_loss_weighting: bool
                If True, gradient balancing will be implemented while
                performing the two-stage training procedure of the IC, BC,
                and interior data losses wrt. the PDE loss. Default is True.

            lambda_ic: float
                The weight on the IC loss. This will be updated if gradient
                balancing is implemented for both the warmup and the two-stage
                training. Default is 1.

            lambda_bc: float
                The weight on the BC loss. This will be updated if gradient
                balancing is implemented for both the warmup and the two-stage
                training. . Default is 1.

            lambda_data: float
                The weight on the interior data loss. This will be updated if 
                gradient balancing is implemented for only the two-stage training.
                Default is 1.

            lambda_pde:
                The weight on the PDE loss. Default is 1.

            lambda_alpha: float
                The weight of the update for lambda_ic and lambda_bc. The weight of the 
                previous value will be (1 - lambda_alpha). Therefore, lambda_alpha
                must be between 0 and 1. Default is 0.9.

            verbosity:
                If equal to 0. No messages tracking the losses will be printed.
                If not equal to 0, this is the incremental number of epochs in 
                n_epochs when messages for the losses will be printed.   
            
        '''
        
        self.lhs_deriv = lhs_deriv
        self.deriv_library = deriv_library
        self.add_const_coeff = add_const_coeff
        self.input_names = input_names
        self.coeffs_input_names = coeffs_input_names
        self.sol_optimizer = sol_optimizer
        self.coeffs_optimizer = coeffs_optimizer
        self.sol_lr_scheduler = sol_lr_scheduler
        self.fit_lr_scheduler = fit_lr_scheduler
        self.n_epochs = n_epochs
        self.sol_warmup_n_epochs = sol_warmup_n_epochs
        self.sol_update_per_epoch = sol_update_per_epoch
        self.coeffs_update_per_epoch = coeffs_update_per_epoch
        self.lambda_pde_warmup_epochs = lambda_pde_warmup_epochs
        self.max_lambda_pde = max_lambda_pde
        self.sol_warmup_loss_weighting = sol_warmup_loss_weighting
        self.fit_loss_weighting = fit_loss_weighting
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_alpha = lambda_alpha
        self.verbosity = verbosity
        self.coeffs_input_idxs = [input_names.index(coeffs_input_name) for coeffs_input_name in coeffs_input_names if coeffs_input_name != 'u']

    def prepare_sol_model(self, sol_model):
        self.sol_model = sol_model

    def prepare_coeffs_model(self, coeffs_model):
        self.coeffs_model = coeffs_model

    def prepare_data(self, X_ic, U_ic, X_bc, U_bc, X_data, U_data, X_col):
        self.X_ic = X_ic
        self.U_ic = U_ic
        self.X_bc = X_bc
        self.U_bc = U_bc
        self.X_data = X_data
        self.U_data= U_data
        self.X_col = X_col

    def _compute_warmup_losses(self, lambda_ic, lambda_bc, lambda_data):
        '''
        Compute the IC, BC, and interior data losses for the warmup.
        
        '''

        # Compute the approximated solution using the solution model.
        u_pred_ic = self.sol_model(self.X_ic)
        u_pred_bc = self.sol_model(self.X_bc)
        u_pred_data = self.sol_model(self.X_data)

        # Compute IC, BC, and interior data losses.
        loss_ic = F.mse_loss(u_pred_ic, self.U_ic)
        loss_bc = F.mse_loss(u_pred_bc, self.U_bc)
        loss_data = F.mse_loss(u_pred_data, self.U_data)
        
        # Implement gradient balancing if needed for IC and BC losses
        # with respect to interior data loss.
        if self.sol_warmup_loss_weighting:
            lambda_ic, lambda_bc = loss_weighting_data_wang(sol_model=self.sol_model,
                                                              loss_ic=loss_ic,
                                                              loss_bc=loss_bc,
                                                              loss_data=loss_data,
                                                              lambda_ic=lambda_ic,
                                                              lambda_bc=lambda_bc,
                                                              lambda_alpha=self.lambda_alpha)
        
        # Calculate the total loss using IC, BC, and interior data losses.
        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_data * loss_data

        return loss_total, loss_ic, loss_bc, loss_data, lambda_ic, lambda_bc, lambda_data
    
    def define_coeffs_input(self, X, U):
        '''
        Helper function to get the input data for the coefficients network(s).
        
        '''

        if 'u' in self.coeffs_input_names:
            coeffs_input = torch.cat([X[:, self.coeffs_input_idxs], U], dim=-1)
        else:
            coeffs_input = X[:, self.coeffs_input_idxs]

        return coeffs_input

    def _compute_losses(self, lambda_ic, lambda_bc, lambda_data, lambda_pde):
        '''
        Compute the losses for the two-stage training procedure using
        IC, BC, interior data, and collocation points.
        
        '''
        
        # Turn on rquires_grad so we can take derivatives for the derivative library.
        self.X_col.requires_grad_(True)

        # Compute the approximated solution using the solution model.
        u_pred_ic = self.sol_model(self.X_ic)
        u_pred_bc = self.sol_model(self.X_bc)
        u_pred_data = self.sol_model(self.X_data)
        u_pred_col = self.sol_model(self.X_col)

        # Compute IC, BC, and interior data losses.
        loss_ic = F.mse_loss(u_pred_ic, self.U_ic)
        loss_bc = F.mse_loss(u_pred_bc, self.U_bc)
        loss_data = F.mse_loss(u_pred_data, self.U_data)

        # Calculate the LHS derivative and the derivatives in the derivative 
        # library.
        lhs_d, D = compute_derivatives(u_pred=u_pred_col,
                                       inputs=self.X_col,
                                       lhs_deriv=self.lhs_deriv,
                                       deriv_library=self.deriv_library,
                                       input_names=self.input_names,
                                       add_ones=self.add_const_coeff)
        
        # Define the inputs to the coefficient network(s).
        coeffs_input = self.define_coeffs_input(self.X_col, u_pred_col)
        # Apply the coefficients network(s) on the inputs.
        coeffs_pred = self.coeffs_model(coeffs_input)
        # Compute the RHS of the PDE.
        pde = torch.sum(D * coeffs_pred, dim=-1, keepdims=True)
        # Compute the PDE loss.
        loss_pde = F.mse_loss(pde, lhs_d)
        
        # Implement gradient balancing if needed for IC, BC, and interior data
        # losses with respect to PDE loss.
        if self.fit_loss_weighting:
            lambda_ic, lambda_bc, lambda_data = loss_weighting_pde_wang(sol_model=self.sol_model,
                                                                          loss_ic=loss_ic,
                                                                          loss_bc=loss_bc,
                                                                          loss_data=loss_data,
                                                                          loss_pde=loss_pde,
                                                                          lambda_ic=lambda_ic,
                                                                          lambda_bc=lambda_bc,
                                                                          lambda_data=lambda_data,
                                                                          lambda_alpha=self.lambda_alpha)

        # Calculate the total loss using IC, BC, interior data, and PDE losses.
        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_data * loss_data + lambda_pde * loss_pde

        return loss_total, loss_ic, loss_bc, loss_data, loss_pde, lambda_ic, lambda_bc, lambda_data, lambda_pde
    
    def sol_warmup_step(self, lambda_ic, lambda_bc):
        '''
        Implements an optimization step for the solution warmup.

        '''

        self.sol_optimizer.zero_grad()
        # Compute IC, BC, and interior data losses.
        loss_total, loss_ic, loss_bc, loss_data, lambda_ic, lambda_bc, lambda_data = self._compute_warmup_losses(lambda_ic, lambda_bc, self.lambda_data)
        
        loss_total.backward()
        self.sol_optimizer.step()
        # Update learning rate using scheduler if needed.
        if self.sol_lr_scheduler is not None:
            self.sol_lr_scheduler.step(loss_total.item())
            
        return loss_total.item(), loss_ic.item(), loss_bc.item(), loss_data.item(), lambda_ic, lambda_bc, lambda_data
    
    def sol_step(self, lambda_ic, lambda_bc, lambda_data, lambda_pde):
        '''
        Implements an optimization step for the stage 1 of the two-stage
        training procedure in which the solution network is trained using
        the IC, BC, interior data and PDE losses with the coefficient newtork
        frozen.

        '''

        self.sol_optimizer.zero_grad()
        # Compute IC, BC, interior data, and PDE losses.
        (loss_total, loss_ic, loss_bc, loss_data, loss_pde, 
         lambda_ic, lambda_bc, lambda_data, lambda_pde) = self._compute_losses(lambda_ic, lambda_bc, lambda_data, lambda_pde)
        
        loss_total.backward()
        self.sol_optimizer.step()
        # Update learning rate using scheduler if needed.
        if self.fit_lr_scheduler is not None:
            self.fit_lr_scheduler.step(loss_total.item())

        return loss_total.item(), loss_ic.item(), loss_bc.item(), loss_data.item(), loss_pde.item(), lambda_ic, lambda_bc, lambda_data, lambda_pde
    
    def coeffs_step(self):
        '''
        Implements an optimization step for the stage 12of the two-stage
        training procedure in which the coefficient network is trained using
        only the PDE loss with the solution network newtork frozen.

        '''

        self.coeffs_optimizer.zero_grad()
        self.X_col.requires_grad_(True)
        # Compute the approximate solution at the collocation points.
        u_pred_col = self.sol_model(self.X_col)

        # Calculate LHS and derivative library derivatives.
        lhs_d, D = compute_derivatives(u_pred=u_pred_col, inputs=self.X_col, lhs_deriv=self.lhs_deriv,
                                       deriv_library=self.deriv_library, input_names=self.input_names,
                                       add_ones=self.add_const_coeff)
        # Define the inputs for the coefficients network(s).
        coeffs_input = self.define_coeffs_input(self.X_col, u_pred_col)
        # Compute the coefficients.
        coeffs_pred = self.coeffs_model(coeffs_input)
        # Compute the RHS of the PDE.
        pde = torch.sum(D * coeffs_pred, dim=-1, keepdims=True)
        # Calculate PDE loss.
        loss = F.mse_loss(pde, lhs_d)
        loss.backward()
        self.coeffs_optimizer.step()

        return loss.item()
    
    def fit(self, sol_model, coeffs_model, X_ic, U_ic, X_bc, U_bc, X_data, U_data, X_col):
        '''
        Fit the solution and coefficients network(s) using IC, BC, interior data,
        and collocation points to discover a PDE using a two-stage procedure. Stage
        1 will train the solution network using the IC, BC, interior data losses with
        the PDE loss being added gradually while the coefficient network is frozen.
        In Stage 2, the solution network will be frozen and the coefficients 
        network(s) will be training using the PDE loss.

        --------
        Arguments
        --------

            sol_model:
                The solution model.
            
            coeffs_model:
                The coefficients model.

            X_ic: torch.Tensor
                The spatio-temporal input IC data.
            
            U_ic: torch.Tensor
                The observed solution at the IC points.

            X_bc: torch.Tensor
                The spatio-temporal input BC data.
            
            U_bc: torch.Tensor
                The observed solution at the BC points.
            
            X_data: torch.Tensor
                The spatio-temporal input interior data.
            
            U_data: torch.Tensor
                The observed solution at the interior points.

            X_data: torch.Tensor
                The spatio-temporal input collocation data.

        '''

        # Prepare models and data.
        self.prepare_sol_model(sol_model=sol_model)
        self.prepare_coeffs_model(coeffs_model=coeffs_model)
        self.prepare_data(X_ic=X_ic, U_ic=U_ic, X_bc=X_bc, U_bc=U_bc, X_data=X_data, U_data=U_data, X_col=X_col)

        # Define lists to track losses.
        total_losses = []
        ic_losses = []
        bc_losses = []
        data_losses = []
        pde_losses = []

        # Warmup training of the solution.
        lambda_ic, lambda_bc, lambda_data = self.lambda_ic, self.lambda_bc, self.lambda_data
        self.sol_model.train()
        for warmup_epoch in range(self.sol_warmup_n_epochs):
            loss_total, loss_ic, loss_bc, loss_data, lambda_ic, lambda_bc, lambda_data = self.sol_warmup_step(lambda_ic, lambda_bc)

            # Track losses during warmup.
            if self.verbosity > 0 and (warmup_epoch + 1) % self.verbosity == 0:
                print(f"[SolWarmup] Ep {warmup_epoch + 1}/{self.sol_warmup_n_epochs} Total={loss_total:.6e}, "
                        f"IC={loss_ic:.6e}, BC={loss_bc:.6e}, Data={loss_data:.6e}, "
                        f"L_IC={lambda_ic:.6e}, L_BC={lambda_bc:.6e}, L_data={lambda_data:.6e}")

        # Two stage training procedure.
        for epoch in range(self.n_epochs):
            # Stage 1: freeze coefficients model and train solution model
            # using IC, BC, interior data, and PDE losses with the last
            # being gradually added with lambda_pde scheduling.
            for param in self.sol_model.parameters():
                param.requires_grad_(True)
            for param in self.coeffs_model.parameters():
                param.requires_grad_(False)
            
            self.sol_model.train()
            self.coeffs_model.eval()
            for _ in range(self.sol_update_per_epoch):
                lambda_pde = lambda_schedule(epoch, self.lambda_pde_warmup_epochs, self.max_lambda_pde)

                (loss_total, loss_ic, loss_bc, loss_data, loss_pde, 
                lambda_ic, lambda_bc, lambda_data, lambda_pde) = self.sol_step(lambda_ic, lambda_bc, lambda_data, lambda_pde)

            # Stage 2: freeze solution model and train coefficient model using
            # PDe loss.
            for param in self.sol_model.parameters():
                param.requires_grad_(False)
            for param in self.coeffs_model.parameters():
                param.requires_grad_(True)
            self.sol_model.eval()
            self.coeffs_model.train()
            for _ in range(self.coeffs_update_per_epoch):
                _ = self.coeffs_step()

            with torch.no_grad():
                total_losses.append(loss_total)
                ic_losses.append(loss_ic)
                bc_losses.append(loss_bc)
                data_losses.append(loss_data)
                pde_losses.append(loss_pde)

            # Track losses.
            if self.verbosity > 0 and (epoch + 1) % self.verbosity == 0:
                print(f"[Train] Ep {epoch + 1}/{self.n_epochs} Total={loss_total:.6e}, "
                        f"IC={loss_ic:.6e}, BC={loss_bc:.6e}, Data={loss_data:.6e}, PDE={loss_pde:.6e}, "
                        f"L_IC={lambda_ic:.6e}, L_BC={lambda_bc:.6e}, L_data={lambda_data:.6e}, L_pde={lambda_pde:.6e}")

        # Save losses and return them.
        res = dict()
        res['total_losses'] = total_losses
        res['ic_losses'] = ic_losses
        res['bc_losses'] = bc_losses
        res['data_losses'] = data_losses
        res['pde_losses'] = pde_losses

        return res