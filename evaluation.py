
import theano
import theano.tensor as T
import numpy as np
import lasagne
from scipy.stats.stats import pearsonr 
import data_handling as data


#Root Mean Square Error
#Input: array of predicted values, array of target values
def rmse(test_pred, targets):
    assert len(test_pred) == len(targets)
    err2 = (test_pred - targets)**2
    rmse = np.sqrt(np.sum(err2)/len(test_pred))
    return rmse
        

#Compute two sided empirical confidence interval. 95% as default        
def conf_int(test_pred, targets, alpha=0.025):
    n = len(test_pred)
    diff = test_pred - targets
    diff_sort = np.sort(diff, axis=0)
    l_ind = int(n*alpha)
    u_ind = int(np.ceil(n-n*alpha))
    l_bound = diff_sort[l_ind]
    u_bound = diff_sort[u_ind]
    return l_bound, u_bound

#Compute symmetric empirical confidence interval. 95% as default    
def sym_conf_int(test_pred, targets, alpha=0.025):
    diff = np.abs(test_pred-targets)
    diff_sort = np.sort(diff, axis=0)
    ind = np.floor(len(test_pred)*alpha)
    print "eval", ind
    return 2*diff_sort[-ind]
    
##Compute relative confidence interval. 95% as default    
def rel_conf_int(test_pred, targets, alpha=0.025, t_0=None, t_N=None):
    l_bound, u_bound = conf_int(test_pred, targets, alpha)
    print l_bound, u_bound
    rel_int = (u_bound-l_bound)/(t_N-t_0)
    return rel_int

 

def relative_error(test_pred, targets, t_0=None, t_N=None):
    if not t_0:
        t_0 = min(targets)
    if not t_N:
        t_N = max(targets)    
    err = rmse(test_pred, targets)
    rel_err = err / (t_N - t_0)
    return rel_err
    
    
def correlation(test_pred, targets):
    return pearsonr(test_pred, targets)
    
     