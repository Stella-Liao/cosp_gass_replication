import numpy as np
import pandas as pd
import geopandas as gpd

import statsmodels.api as sm

import scipy
from spglm import family
from scipy.optimize import dual_annealing

from copy import deepcopy
from smoother3 import ConstantTerm, LinearTerm, DistanceSmoothing, KernelSmoothing

class GASS_iwls:
    def __init__(self, y, *terms, family = 'Gaussian', constant = True):
        self.y = y
        self.family = family.lower() # by default, Gaussian, so far only Gaussian
        self.terms = terms 
        self.constant = constant 
      
        self._initialize()

    def _initialize(self):
        init_X_mat = [] 
        X0 = [] 
        
        instances = []
        int_mask = [] 
        init_scales = [] 
        scale_bounds = []
      
        if self.constant == True:
            constant = ConstantTerm(self.y.shape[0])
            init_X_mat.append(constant.X)
            X0.append(constant.X)
        
        for _, term in enumerate(self.terms):

            if isinstance(term, LinearTerm):
                init_X_mat.append(term.X)
                X0.append(term.X)
              
            elif isinstance(term, DistanceSmoothing):
                xs = term.get_cached(term.initial_scale)
                init_X_mat.append(xs)
                init_scales.append(term.initial_scale)
                scale_bounds.append((term.lower_bound, term.upper_bound))
                int_mask.append(False) 
                instances.append(term)
                
            elif isinstance(term, KernelSmoothing):
                xs = term.get_cached(term.initial_scale)
                init_X_mat.append(xs)
                init_scales.append(term.initial_scale)
                scale_bounds.append((term.lower_bound, term.upper_bound))
                int_mask.append(True) 
                instances.append(term)
                
            else:
                raise ValueError(f"Unsupported term type: {type(term)}")

        self.init_X = np.hstack(init_X_mat)
        self.init_betas = self._iwls_gaussian(self.init_X, self.y)
        self.X0 = None if not X0 else np.hstack(X0)
        self.n, self.k = self.init_X.shape
        self.instances = instances
        self.int_mask = int_mask
        self.init_scales = init_scales
        self.scale_bounds = scale_bounds
           
    def fit(self, input_y = None, max_iter=100, tol=1e-6, printed = False):
        
        y = self.y if input_y is None else input_y.copy()
        X0 = self.X0
        instances = self.instances
        n, k = self.n, self.k
         
        betas = self.init_betas 
        betas_new = betas  
        
        scales = self.init_scales 
        scale_history = [scales]
        scale_bounds = self.scale_bounds
        aic_history = []
        
        int_mask = self.int_mask
        
        n_iter = 0
        
        for n_iter in range(max_iter):
        
            def _compute_Xs(scales):
                Xs = []
                rounded_scales = []

                for mask, instance, scale in zip(int_mask, instances, scales):
                    rounded_scale = int(scale) if mask else np.round(scale, 3)
                    Xs.append(instance.get_cached(rounded_scale))
                    rounded_scales.append(rounded_scale)
                
                return np.hstack(Xs), np.array(rounded_scales)
            
            def _aic_func(scales):
                Xs, _ = _compute_Xs(scales)
                X_temp = Xs if X0 is None else np.column_stack((X0, Xs))
                
                residual = y - X_temp @ betas_new
                rss = np.sum(residual ** 2)
                aic = n * np.log(rss / n) + 2 * k
                
                return aic

            result = dual_annealing(_aic_func, bounds=scale_bounds, x0 = scale_history[-1])
            
            Xs_new, scales_new = _compute_Xs(result.x)
            
            scale_history.append(scales_new)
            aic_history.append(result.fun)
            
            X_new = Xs_new if X0 is None else np.column_stack((X0, Xs_new))
            betas_new = self._iwls_gaussian(X_new, y)
            
            if printed:
                print(f"Iteration {n_iter+1}: betas_new = {betas_new}, scales_new = {scales_new}, AIC = {current_aic:.2f}")

            if np.linalg.norm(betas_new - betas) < tol: 
                break
            
            betas = betas_new

        self.coefficients = betas_new
        self.scales = scales_new
        self.final_X = X_new
        
        pass
    
    def _iwls_gaussian(self, X, y, max_iter=100, tol=1e-6):
        n, p = X.shape
        beta = np.zeros(p)
        for i in range(max_iter):
            eta = X @ beta
            mu = eta
            W = np.eye(n)
            z = y
            XTWX = X.T @ W @ X
            XTWz = X.T @ W @ z
            beta_new = np.linalg.solve(XTWX, XTWz)
            if np.linalg.norm(beta_new - beta) < tol:
                break
            beta = beta_new
        return beta
    
    def _family_handler(self, y):
            """Handles all family-specific operations."""
        
            supported_families = {
                'gaussian': family.Gaussian()
            }

            fam = supported_families.get(self.family)
            if fam is None:
                raise ValueError(f"Unsupported family: {self.family}")

            # Gaussian
            if isinstance(fam, family.Gaussian):
                def init_gaussian():
                    return None, None

                def adjust_response_gaussian(v, mu):
                    w = np.ones(len(y)).reshape(-1, 1)  
                    z = y.reshape(-1, 1)  
                    return w, z

                def update_statistical_weights_gaussian(X, betas):
                    return None, None

                def infer_gaussian(X, betas):
                    n, k = X.shape
                    yhat = np.dot(X, betas).flatten()
                    residuals = y - yhat
                    s2 = np.sum(residuals ** 2) / (X.shape[0] - X.shape[1])  # Variance

                    var_beta = s2 * np.linalg.inv(X.T @ X).diagonal()
                    std_err = np.sqrt(var_beta)
                    r2 = 1 - (np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))
                    logLm = -n/2 * (1 + np.log(2*np.pi)) - n/2 * np.log(s2)
                    aic =  2*k - 2*logLm

                    return {
                        'fitted_y': yhat,
                        'residuals': residuals,
                        'std_err': std_err,
                        'R_squared': r2,
                        'log_likelihood': logLm,
                        'AIC': aic
                    }

                return {'init': init_gaussian, 'adjust': adjust_response_gaussian, 'update': update_statistical_weights_gaussian, 'infer': infer_gaussian}
            
    def _calculate_CI_betas(self, betas, se_beta, n, k, dist="t"):
        if dist == "t":
            critical_value = scipy.stats.t.ppf(1 - 0.05 / 2, df = n - k)
        elif dist == "z":
            critical_value = scipy.stats.norm.ppf(1 - 0.05 / 2)
        else:
            raise ValueError("Invalid distribution type for confidence interval calculation.")
        
        coefs_lower = betas.flatten() - critical_value * se_beta
        coefs_upper = betas.flatten() + critical_value * se_beta
        return list(zip(coefs_lower, coefs_upper))
    
    def inference(self):
        # Ensure the family type
        if self.family is None:
            raise ValueError("The `family` parameter must be specified (e.g., 'Gaussian' or 'Poisson').")

        family_ops = self._family_handler(self.y)
        infer_results = family_ops['infer'](self.final_X, self.coefficients)
        n, k = self.final_X.shape
        
        # Store results
        self.fitted_y = infer_results['fitted_y']
        self.residuals = infer_results['residuals']
        self.std_err = infer_results['std_err']
        self.log_likelihood = infer_results['log_likelihood']
        self.AIC = infer_results['AIC']

        # Compute confidence intervals of betas
        self.CI_betas = self._calculate_CI_betas(self.coefficients, self.std_err, n, k, dist="t" if self.family == "gaussian" else "z")
        
        # Compute p-values
        self.tvals = self.coefficients.flatten() / self.std_err
        self.pvals = 2 * (1 - scipy.stats.t.cdf(np.abs(self.tvals), df = n - k)) if self.family == 'gaussian' else 2 * (1 - scipy.stats.norm.cdf(np.abs(self.tvals)))

        # Store results for differnet families
        if self.family == 'gaussian':
            self.R_squared = infer_results['R_squared']
    
    def calculate_AWCI_scales(self, level = 0.95):
        
        instances = self.instances
        AWCI_scales = []
        int_mask = self.int_mask
        
        # only consider gaussain
        w = 1
        wy = w * self.y
        wx = self.final_X
        betas = self.coefficients
        n, k = self.final_X.shape
        
        for tidx, tscale in enumerate(self.scales):
            
            tinstance = instances[tidx]
            is_int = int_mask[tidx]
            x0_cols = self.X0.shape[1] if self.X0 is not None else 0
            tidx = int(tidx + x0_cols)
            # tidx = int(tidx + self.X0.shape[1]) # get the true index in whole Xs
            
            # create an array of candidate scales
            tscale_b4 = np.arange(tinstance.lower_bound, tscale, tinstance.CI_step)
            tscale_af = np.arange(tscale, tinstance.upper_bound, tinstance.CI_step) 
            tscale_candidates = np.hstack((tscale_b4, tscale_af)).flatten()
            
            tscale_aics = []
            cols = np.arange(k)
            
            for scale in tscale_candidates:
            
                rounded_scale = int(scale) if is_int else np.round(scale, 3)
                twx = np.hstack((wx[:, cols!= tidx], tinstance.get_cached(rounded_scale) * w))
                tbetas = self._iwls_gaussian(twx, wy)
                
                residual = wy - twx @ tbetas
                rss = np.sum(residual ** 2)
                aic = n * np.log(rss / n) + 2 * k
                tscale_aics.append((scale, aic))
                
            tscale_awdf = pd.DataFrame(tscale_aics, columns=['scale', 'AIC'])  

            minAIC = np.min(tscale_awdf.AIC)
            deltaAICs = tscale_awdf.AIC - minAIC
            awsum = np.sum(np.exp(-0.5 * deltaAICs))
            tscale_awdf = tscale_awdf.assign(AW = np.exp(-0.5 * deltaAICs)/awsum)
            tscale_awdf = tscale_awdf.sort_values(by = 'AW',ascending=False)
            tscale_awdf = tscale_awdf.assign(cumAW = tscale_awdf.AW.cumsum())

            index = len(tscale_awdf[tscale_awdf.cumAW < level]) + 1
            tscale_min = tscale_awdf.iloc[:index,:].scale.min()
            tscale_max = tscale_awdf.iloc[:index,:].scale.max()
            
            AWCI_scales.append((round(tscale_min, 4), round(tscale_max,4)))
            self.AWCI_scales = AWCI_scales
            
        pass