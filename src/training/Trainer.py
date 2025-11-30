from utils.loss_weighting import *
from utils.derivatives import compute_derivatives
import torch.nn.functional as F

class Trainer:
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
        u_pred_ic = self.sol_model(self.X_ic)
        u_pred_bc = self.sol_model(self.X_bc)
        u_pred_data = self.sol_model(self.X_data)

        loss_ic = F.mse_loss(u_pred_ic, self.U_ic)
        loss_bc = F.mse_loss(u_pred_bc, self.U_bc)
        loss_data = F.mse_loss(u_pred_data, self.U_data)
        
        if self.sol_warmup_loss_weighting:
            lambda_ic, lambda_bc = loss_weighting_data_wang(sol_model=self.sol_model,
                                                              loss_ic=loss_ic,
                                                              loss_bc=loss_bc,
                                                              loss_data=loss_data,
                                                              lambda_ic=lambda_ic,
                                                              lambda_bc=lambda_bc,
                                                              lambda_alpha=self.lambda_alpha)

        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_data * loss_data

        return loss_total, loss_ic, loss_bc, loss_data, lambda_ic, lambda_bc, lambda_data
    
    def define_coeffs_input(self, X, U):
        if 'u' in self.coeffs_input_names:
            coeffs_input = torch.cat([X[:, self.coeffs_input_idxs], U], dim=-1)
        else:
            coeffs_input = X[:, self.coeffs_input_idxs]

        return coeffs_input

    def _compute_losses(self, lambda_ic, lambda_bc, lambda_data, lambda_pde):
        self.X_col.requires_grad_(True)

        u_pred_ic = self.sol_model(self.X_ic)
        u_pred_bc = self.sol_model(self.X_bc)
        u_pred_data = self.sol_model(self.X_data)
        u_pred_col = self.sol_model(self.X_col)

        loss_ic = F.mse_loss(u_pred_ic, self.U_ic)
        loss_bc = F.mse_loss(u_pred_bc, self.U_bc)
        loss_data = F.mse_loss(u_pred_data, self.U_data)

        lhs_d, D = compute_derivatives(u_pred=u_pred_col,
                                       inputs=self.X_col,
                                       lhs_deriv=self.lhs_deriv,
                                       deriv_library=self.deriv_library,
                                       input_names=self.input_names,
                                       add_ones=self.add_const_coeff)
        
        coeffs_input = self.define_coeffs_input(self.X_col, u_pred_col)
        coeffs_pred = self.coeffs_model(coeffs_input)
        pde = torch.sum(D * coeffs_pred, dim=-1, keepdims=True)

        loss_pde = F.mse_loss(pde, lhs_d)

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

        loss_total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_data * loss_data + lambda_pde * loss_pde

        return loss_total, loss_ic, loss_bc, loss_data, loss_pde, lambda_ic, lambda_bc, lambda_data, lambda_pde
    
    def sol_warmup_step(self, lambda_ic, lambda_bc):
        self.sol_optimizer.zero_grad()
        loss_total, loss_ic, loss_bc, loss_data, lambda_ic, lambda_bc, lambda_data = self._compute_warmup_losses(lambda_ic, lambda_bc)
        
        loss_total.backward()
        self.sol_optimizer.step()
        if self.sol_lr_scheduler is not None:
            self.sol_lr_scheduler.step(loss_total.item())
            
        return loss_total.item(), loss_ic.item(), loss_bc.item(), loss_data.item(), lambda_ic, lambda_bc, lambda_data
    
    def sol_step(self, lambda_ic, lambda_bc, lambda_data, lambda_pde):
        self.sol_optimizer.zero_grad()
        (loss_total, loss_ic, loss_bc, loss_data, loss_pde, 
         lambda_ic, lambda_bc, lambda_data, lambda_pde) = self._compute_losses(lambda_ic, lambda_bc, lambda_data, lambda_pde)
        
        loss_total.backward()
        self.sol_optimizer.step()
        if self.fit_lr_scheduler is not None:
            self.fit_lr_scheduler.step(loss_total.item())

        return loss_total.item(), loss_ic.item(), loss_bc.item(), loss_data.item(), loss_pde.item(), lambda_ic, lambda_bc, lambda_data, lambda_pde
    
    def coeffs_step(self):

        self.coeffs_optimizer.zero_grad()
        self.X_col.requires_grad_(True)
        u_pred_col = self.sol_model(self.X_col)

        lhs_d, D = compute_derivatives(u_pred=u_pred_col, inputs=self.X_col, lhs_deriv=self.lhs_deriv,
                                       deriv_library=self.deriv_library, input_names=self.input_names,
                                       add_ones=self.add_const_coeff)
        
        coeffs_input = self.define_coeffs_input(self.X_col, u_pred_col)
        coeffs_pred = self.coeffs_model(coeffs_input)
        pde = torch.sum(D * coeffs_pred, dim=-1, keepdims=True)

        loss = F.mse_loss(pde, lhs_d)
        loss.backward()
        self.coeffs_optimizer.step()

        return loss.item()
    
    def fit(self, sol_model, coeffs_model, X_ic, U_ic, X_bc, U_bc, X_data, U_data, X_col):
        self.prepare_sol_model(sol_model=sol_model)
        self.prepare_coeffs_model(coeffs_model=coeffs_model)
        self.prepare_data(X_ic=X_ic, U_ic=U_ic, X_bc=X_bc, U_bc=U_bc, X_data=X_data, U_data=U_data, X_col=X_col)

        total_losses = []
        ic_losses = []
        bc_losses = []
        data_losses = []
        pde_losses = []

        lambda_ic, lambda_bc, lambda_data = self.lambda_ic, self.lambda_bc, self.lambda_data
        self.sol_model.train()
        for warmup_epoch in range(self.sol_warmup_n_epochs):
            loss_total, loss_ic, loss_bc, loss_data, lambda_ic, lambda_bc, lambda_data = self.sol_warmup_step(lambda_ic, lambda_bc)

            if self.verbosity > 0 and (warmup_epoch + 1) % self.verbosity == 0:
                print(f"[SolWarmup] Ep {warmup_epoch + 1}/{self.sol_warmup_n_epochs} Total={loss_total:.6e}, "
                        f"IC={loss_ic:.6e}, BC={loss_bc:.6e}, Data={loss_data:.6e}, "
                        f"L_IC={lambda_ic:.6e}, L_BC={lambda_bc:.6e}, L_data={lambda_data:.6e}")

        for epoch in range(self.n_epochs):
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
                
                with torch.no_grad():
                    total_losses.append(loss_total)
                    ic_losses.append(loss_ic)
                    bc_losses.append(loss_bc)
                    data_losses.append(loss_data)
                    pde_losses.append(loss_pde)

            for param in self.sol_model.parameters():
                param.requires_grad_(False)
            for param in self.coeffs_model.parameters():
                param.requires_grad_(True)
            self.sol_model.eval()
            self.coeffs_model.train()
            for _ in range(self.coeffs_update_per_epoch):
                _ = self.coeffs_step()

            if self.verbosity > 0 and (epoch + 1) % self.verbosity == 0:
                print(f"[Train] Ep {epoch + 1}/{self.n_epochs} Total={loss_total:.6e}, "
                        f"IC={loss_ic:.6e}, BC={loss_bc:.6e}, Data={loss_data:.6e}, PDE={loss_pde:.6e}, "
                        f"L_IC={lambda_ic:.6e}, L_BC={lambda_bc:.6e}, L_data={lambda_data:.6e}, L_pde={lambda_pde:.6e}")

        res = dict()
        res['total_losses'] = total_losses
        res['ic_losses'] = ic_losses
        res['bc_losses'] = bc_losses
        res['data_losses'] = data_losses
        res['pde_losses'] = pde_losses

        return res