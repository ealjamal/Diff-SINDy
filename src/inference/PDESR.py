from pysr import PySRRegressor
import torch
import numpy as np

class PDESR:
    def __init__(self, trainer, **sr_kwargs):
        self.trainer = trainer
        if 'output_dir' in sr_kwargs.keys():
          sr_kwargs['output_dir'] = None
        self.sr_model = PySRRegressor(**sr_kwargs)
    
    def fit(self, X, U=None, save_dir=None):
        terms_res = dict()
        with torch.no_grad():
            if U is None:
                U = self.trainer.sol_model(X).detach()
            else:
                U = U.detach()
            
            coeffs_input = self.trainer.define_coeffs_input(X, U).detach()
            coeffs_pred = self.trainer.coeffs_model(coeffs_input).detach().cpu().numpy()
        
        terms = ["const"] + self.trainer.deriv_library if self.trainer.add_const_coeff else self.trainer.deriv_library
        for i, term in enumerate(terms):
            print(f"\n\033[34mFitting SR on {term} coefficient...\033[0m")
            coeff_sr = self.sr_model.fit(coeffs_input.detach().cpu().numpy(), coeffs_pred[:, i], 
                                         variable_names=self.trainer.coeffs_input_names)
            terms_res[f"{term}"] = coeff_sr.equations_.to_dict()
            print(f"\n\033[32mFinished SR on {term} coefficient!\033[0m")

        if save_dir is not None:
            np.savez(save_dir, **terms_res)

        return terms_res