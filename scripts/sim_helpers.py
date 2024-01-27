
import cvxpy as cp
import numpy as np
from functools import partial, lru_cache
from scipy.optimize import linprog
from sklearn.metrics.pairwise import pairwise_kernels
from typing import Callable

 

 

def phi_fn_x(x : np.ndarray):
    return x
def phi_fn_intercept(x):
    return np.ones((x.shape[0], 1))
 




def run_conditional_kernel(x_calib : np.ndarray,
            y_calib : np.ndarray,
            x_test : np.ndarray,
            predictor : callable,
            alpha: float = 0.05,
            lambd: float = -1):
              infinite_params = {'kernel': 'rbf', 'gamma': 4, 'lambda': 0.005}
              phi_fn = phi_fn_x

              score_fn = lambda x, y : np.abs(y - predictor(x))
              score_inv_fn = lambda s, x : [predictor(x) - s, predictor(x) + s]


              if(lambd < 0):
                allLosses, radii = runCV(x_calib,score_fn(x_calib, y_calib),infinite_params['kernel'],infinite_params['gamma'], alpha,10,
                                 10, 10000, 50, phi_fn(x_calib))
                print(allLosses)
                selectedRadius = radii[np.argmin(allLosses)]
                print(selectedRadius)
                infinite_params['lambda'] = 1/selectedRadius
              else: 
                infinite_params['lambda'] = lambd

              print(infinite_params['lambda'])
              cond_conf = CondConf(score_fn, phi_fn, infinite_params = infinite_params)
              cond_conf.setup_problem(x_calib, y_calib)
              
              n_test = len(x_test)

              lbs = np.zeros((n_test,))
              ubs = np.zeros((n_test,))

              lbs_r = np.zeros((n_test,))
              ubs_r = np.zeros((n_test,))
               
               

              i = 0
              for x_t in x_test:
                print(i)
                x_t = x_t.reshape(1, -1)
                try: 
                  res = cond_conf.predict(1 - alpha, x_t, score_inv_fn, exact=False, randomize=True)
                  lbs[i] = res[0][0,0]
                  ubs[i] = res[1][0,0]
                except:
                  print("The solver didn't work")
                  lbs[i] = np.nan
                  ubs[i] = np.nan
                 

                i += 1
    
    
              #out = cond_conf.predict(alpha / 2, x_t, score_inv_fn_lb, exact=True, randomize=True)
              return lbs, ubs, predictor(x_test)
            
            
            




def run_conditional_exact(x_calib : np.ndarray,
            y_calib : np.ndarray,
            x_test : np.ndarray,
            predictor : callable,
            phi_fn : callable,
            alpha: float = 0.05):
              infinite_params = {'kernel': None, 'gamma': None, 'lambda': 0}

              score_fn = lambda x, y : np.abs(y - predictor(x))
              score_inv_fn = lambda s, x : [predictor(x) - s, predictor(x) + s]


           
              cond_conf = CondConf(score_fn, phi_fn, infinite_params = infinite_params)
              cond_conf.setup_problem(x_calib, y_calib)
              
              n_test = len(x_test)

              lbs = np.zeros((n_test,))
              ubs = np.zeros((n_test,))

              lbs_r = np.zeros((n_test,))
              ubs_r = np.zeros((n_test,))
               
               

              i = 0
              for x_t in x_test:
                print(i)
                x_t = x_t.reshape(1, -1)
                try: 
                  res = cond_conf.predict(1 - alpha, x_t, score_inv_fn, exact=True, randomize=True)
                  lbs[i] = res[0][0,0]
                  ubs[i] = res[1][0,0]
                except:
                  print("The solver didn't work")
                  lbs[i] = np.nan
                  ubs[i] = np.nan

                i += 1
    
    
              return lbs, ubs, predictor(x_test)
            
            
            
