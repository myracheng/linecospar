
"""
These are functions used in preference GP learning, for learning a Bayesian 
preference model given preference data and drawing samples from this model.

In this file, the sigmoidal link function is used to capture the user's pref-
erence noise.
"""

import numpy as np
from scipy.optimize import minimize

def advance(posterior_model, num_samples, cov_scale = 1):
    """
    Draw a specified number of samples from the preference GP Bayesian model 
    posterior.
    
    Inputs:
        1) posterior_model: this is the model posterior, represented as a 
           dictionary of the form {'mean': post_mean, 'cov_evecs': evecs, 
           'cov_evals': evals}; post_mean is the posterior mean, a length-n 
           NumPy array in which n is the number of points over which the 
           posterior is to be sampled. cov_evecs is an n-by-n NumPy array in 
           which each column is an eigenvector of the posterior covariance,
           and evals is a length-n array of the eigenvalues of the posterior 
           covariance.
           
        2) num_samples: the number of samples to draw from the posterior; a 
           positive integer.
           
        3) cov_scale: parameter between 0 and 1; this is multiplied to the 
           posterior standard deviation, so values closer to zero result in 
           less exploration.
           
    Outputs:
        1) A num_samples-length NumPy array, in which each element is the index
           of a sample.
        2) A num_samples-length list, in which each entry is a sampled reward
           function. Each reward function sample is a length-n vector (see 
           above for definition of n).
    
    """
    
    samples = np.empty(num_samples)    # To store the sampled actions
    
    # Unpack model posterior:
    mean = posterior_model['mean']
    cov_evecs = posterior_model['cov_evecs']
    cov_evals = posterior_model['cov_evals']
    
    num_features = len(mean)
    
    R_models = []       # To store the sampled reward functions
    
    # Draw the samples:
    for i in range(num_samples):
    
        # Sample reward function from GP model posterior:
        X = np.random.normal(size = num_features)
        R = mean + cov_scale * cov_evecs @ np.diag(np.sqrt(cov_evals)) @ X
        
        R = np.real(R)
        
        samples[i] = np.argmax(R) # Find where reward function is maximized
        
        R_models.append(R)        # Store sampled reward function
        
    return samples.astype(int), R_models
    

def feedback(data, labels, GP_prior_cov_inv, preference_noise, cov_scale = 1,
             r_init = []):
    """
    Function for updating the GP preference model given data.
    
    Inputs:
        1) data: num_data_points-by-data_point_dimensionality NumPy array
        2) labels: num_data_points NumPy array (all elements should be zeros
           and ones)
        3) GP_prior_cov_inv: n-by-n NumPy array, where n is the number of 
           points over which the posterior is to be sampled 
        4) preference_noise: positive scalar parameter. Higher values indicate
           larger amounts of noise in the expert preferences.
        5) (Optional) cov_scale: parameter between 0 and 1; this is multiplied 
           to the posterior standard deviation when sampling in the advance 
           function, so values closer to zero result in less exploration.
        6) (Optional) initial guess for convex optimization; length-n NumPy
           array when specified.
               
    Output: the updated model posterior, represented as a dictionary of the 
           form {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals};
           post_mean is the posterior mean, a length-n NumPy array in which n
           is the number of points over which the posterior is to be sampled.
           cov_evecs is an n-by-n NumPy array in which each column is an
           eigenvector of the posterior covariance, and evals is a length-n 
           array of the eigenvalues of the posterior covariance.
    
    """   
    num_features = GP_prior_cov_inv.shape[0]

    # Solve convex optimization problem to obtain the posterior mean reward 
    # vector via Laplace approximation:    
    if r_init == []:
        r_init = np.zeros(num_features)    # Initial guess

    res = minimize(preference_GP_objective, r_init, args = (data, labels, 
                   GP_prior_cov_inv, preference_noise), method='L-BFGS-B', 
                   jac=preference_GP_gradient)
    
    # The posterior mean is the solution to the optimization problem:
    post_mean = res.x

    if cov_scale > 0: # Calculate eigenvectors/eigenvalues of covariance matrix

        # Obtain inverse of posterior covariance approximation by evaluating the
        # objective function's Hessian at the posterior mean estimate:
        post_cov_inverse = preference_GP_hessian(post_mean, data, labels, 
                       GP_prior_cov_inv, preference_noise) 
    
        # Calculate the eigenvectors and eigenvalues of the inverse posterior 
        # covariance matrix:
        evals, evecs = np.linalg.eigh(post_cov_inverse)
    
        # Invert the eigenvalues to get the eigenvalues corresponding to the 
        # covariance matrix:
        evals = 1 / evals
        
    else:   # cov_scale = 0; evecs/evals not used in advance function.
    
        evecs = np.eye(num_features)
        evals = np.zeros(num_features)
    
    # Return the model posterior:
    best_idx = np.argmax(post_mean) 

    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals, 'best_idx': best_idx}

    
def preference_GP_objective(f, data, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the optimization objective function for finding the posterior 
    mean of the GP preference model (at a given point); the posterior mean is 
    the minimum of this (convex) objective function.
    
    Inputs:
        1) f: the "point" at which to evaluate the objective function. This is
           a length-n vector, where n is the number of points over which the 
           posterior is to be sampled.
        2)-5): same as the descriptions in the feedback function. 
        
    Output: the objective function evaluated at the given point (f).
    """
    
    
    obj = 0.5 * f @ GP_prior_cov_inv @ f
    
    num_samples = data.shape[0]
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = data[i, :].astype(int)   # Data points queried in this sample
        label = labels[i]
        
        if data_pts[0] == data_pts[1] or label == 0.5:
            continue
        
        label = int(label)
        
        z = (f[data_pts[label]] - f[data_pts[1 - label]]) / preference_noise
        obj -= np.log(sigmoid(z))
        
    return obj


def preference_GP_gradient(f, data, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the gradient of the optimization objective function for finding 
    the posterior mean of the GP preference model (at a given point).
    
    Inputs:
        1) f: the "point" at which to evaluate the gradient. This is a length-n
           vector, where n is the number of points over which the posterior 
           is to be sampled.
        2)-5): same as the descriptions in the feedback function. 
        
    Output: the objective function's gradient evaluated at the given point (f).
    """
    
    grad = GP_prior_cov_inv @ f    # Initialize to 1st term of gradient
    
    num_samples = data.shape[0]
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = data[i, :].astype(int)   # Data points queried in this sample
        if labels[i] < 0.5:
            label = 0
        else:
            label = 1
        # label = int(labels[i])
        # print(label)
        if data_pts[0] == data_pts[1] or label == 0.5:
            continue
        
        s_pos = data_pts[label]
        s_neg = data_pts[1 - label]
        
        z = (f[s_pos] - f[s_neg]) / preference_noise
        
        value = (sigmoid_der(z) / sigmoid(z)) / preference_noise
        
        grad[s_pos] -= value
        grad[s_neg] += value
        
    return grad

def preference_GP_hessian(f, data, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the Hessian matrix of the optimization objective function for 
    finding the posterior mean of the GP preference model (at a given point).
    
    Inputs:
        1) f: the "point" at which to evaluate the Hessian. This is
           a length-n vector, where n is the number of points over which the 
           posterior is to be sampled.
        2)-5): same as the descriptions in the feedback function. 
        
    Output: the objective function's Hessian matrix evaluated at the given 
            point (f).
    """
    
    num_samples = data.shape[0]
    
    Lambda = np.zeros(GP_prior_cov_inv.shape)
    
    for i in range(num_samples):   # Go through each pair of data points
        
        data_pts = data[i, :].astype(int)  # Data points queried in this sample
        label = int(labels[i])

        if data_pts[0] == data_pts[1] or label == 0.5:
            continue
        
        s_pos = data_pts[label]
        s_neg = data_pts[1 - label]
        
        z = (f[s_pos] - f[s_neg]) / preference_noise
        
        sigm = sigmoid(z)
        # print(type((sigmoid_der / sigm)**2))
        value = (-sigmoid_2nd_der(z) / sigm + (sigmoid_der(z) / sigm)**2) / (preference_noise**2)
        
        Lambda[s_pos, s_pos] += value
        Lambda[s_neg, s_neg] += value
        Lambda[s_pos, s_neg] -= value
        Lambda[s_neg, s_pos] -= value
    
    return GP_prior_cov_inv + Lambda
    

def sigmoid(x):
    """
    Evaluates the sigmoid function at the specified value.
    Input: x = any scalar
    Output: the sigmoid function evaluated at x.
    """
    
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    """
    Evaluates the sigmoid function's derivative at the specified value.
    Input: x = any scalar
    Output: the sigmoid function's derivative evaluated at x.
    """
    
    return np.exp(-x) / (1 + np.exp(-x))**2

def sigmoid_2nd_der(x):
    """
    Evaluates the sigmoid function's 2nd derivative at the specified value.
    Input: x = any scalar
    Output: the sigmoid function's 2nd derivative evaluated at x.
    """
    
    return (-np.exp(-x) + np.exp(-2 * x)) / (1 + np.exp(-x))**3
    