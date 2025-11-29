from pysr import PySRRegressor
import torch

class PDESR:
    def __init__(self, trainer, **sr_kwargs):
        self.trainer = trainer
        self.sr_model = PySRRegressor(**sr_kwargs)
    
    def fit(self, X, U=None):
        with torch.no_grad():
            if U is None:
                U = self.trainer.sol_model(X).detach()
            else:
                U = U.detach()
            
            coeffs_input = self.trainer.define_coeffs_input(X, U).detach()
            coeffs_pred = self.trainer.coeffs_model(coeffs_input).detach().cpu().numpy()

        self.sr_model.fit(coeffs_input.detach().cpu().numpy(), coeffs_pred)

        return self.sr_model