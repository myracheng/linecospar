
"""
Use CoSpar framework to optimize over Hartmann.

This version: no buffer (n > 1, b = 0); no coactive feedback. As this version 
does not use buffers or coactive feedback, it is actually the Self-Sparring 
algorithm, with the GP preference model of Chu and Ghahramani (2005) used to 
model the latent reward function.
"""

import numpy as np
import os
import scipy.io as io
import itertools
import time
from utils import *

from Preference_GP_learning import advance, feedback

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
rcParams.update({'font.size': 24})

fig_num = 0

plt.close('all')

# Number of times to sample the GP model on each iteration. The algorithm 
# will obtain preferences between all non-overlapping groups of num_samples 
# samples.
num_samples = 3

run_nums = np.arange(10)     # Repeat test once for each number in this list.

# GP model hyperparameters. The underlying reward function has a Gaussian
# process prior with a squared exponential kernel.
signal_variance = 0.0001   # Gaussian process amplitude parameter
preference_noise = 0.01    # How noisy are the user's preferences?
GP_noise_var = 1e-5        # GP model noise--need at least a very small
                           # number to ensure that the covariance matrix
                           #  is invertible.

num_trials = 1000     # Total number of posterior samples/trials
num_iterations = int(np.ceil(num_trials / num_samples))

# Folder in which to save the results.
save_folder = 'Results/baseline_hartmann_%s/' % str(time.time())[-3:]

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  

# Load points in the grid over which objective functions were sampled.
lower_bound = 0
upper_bound = 1
state_dim = 4
points_per_dimension = 10 #m
num_visited_points = 0 #N
space_range = np.linspace(lower_bound, upper_bound, points_per_dimension)
lengthscales = [0.15] * state_dim          # Larger values = smoother reward function

def objective_function(pt):
    """3d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
    """
    alpha = [1.00, 1.20, 3.00, 3.20]
    #Hartmann-6
    # A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
    #               [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
    #               [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
    #               [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    # P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
    #                        [2329, 4135, 8307, 3736, 1004, 9991],
    #                        [2348, 1451, 3522, 2883, 3047, 6650],
    #                        [4047, 8828, 8732, 5743, 1091, 381]])
   
    #Hartmann-3
    A = np.array([[3.0, 10.0, 30.0],
                    [0.1, 10.0, 35.0],
                    [3.0, 10.0, 30.0],
                    [0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                            [4699, 4387, 7470],
                            [1090, 8732, 5547],
                            [381, 5743, 8828]])
    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(state_dim):
            internal_sum += A[i, j] * (pt[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)

    
    return external_sum

points_to_sample = np.array(np.meshgrid(space_range,space_range,space_range,space_range)).T.reshape(-1,state_dim)

# Get 1) dimension of input space and 2) number of points in objective function
# grid (also the number of points we will jointly sample in each posterior sample)
num_pts_sample = len(points_to_sample)          

# Instantiate the prior covariance matrix, using a squared exponential
# kernel in each dimension of the input space:
GP_prior_cov =  signal_variance * np.ones((num_pts_sample, num_pts_sample))   
print(np.shape(points_to_sample))
for i in range(num_pts_sample):

    pt1 = points_to_sample[i,:]
    for j in range(num_pts_sample):
        
        pt2 = points_to_sample[j,:]
        for dim in range(state_dim):
            lengthscale = lengthscales[dim]
            
            if lengthscale > 0:
                GP_prior_cov[i, j] *= np.exp(-0.5 * ((pt2[dim] - pt1[dim]) / \
                            lengthscale)**2)
                
            elif lengthscale == 0 and pt1[dim] != pt2[dim]:
                
                GP_prior_cov[i, j] = 0
 
GP_prior_cov += GP_noise_var * np.eye(num_pts_sample)
       
GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix

# List of all pairs of samples between which to obtain pairwise preferences 
# (there are (num_samples choose 2) of these):
preference_queries = list(itertools.combinations(np.arange(num_samples), 2))
num_pref = len(preference_queries)    # Pairwise preferences per iteration

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):

    # Load objective function to use in this experimental repetition.
        
    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * num_iterations * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref * num_iterations, 2)).astype(int)
    labels = np.empty((num_pref * num_iterations, 2))

    # Also store objective function values (for diagnostic purposes only--the 
    # learning algorithm cannot see this):
    objective_values = []

    pref_count = 0  # Keeps track of how many preferences are in the dataset

    # In each iteration, we sample num_samples points from the model posterior, 
    # and obtain all possible pairwise preferences between these points.   
    for it in range(num_iterations):
       
        # Print status:
        print('Run %i of %i, iteration %i of %i' % (i + 1, len(run_nums), 
            it + 1, num_iterations))
        
        # Preference data observed so far (used to train GP preference model):
        X = data_pt_idxs[: pref_count, :]
        y = labels[: pref_count, 1]
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)    
        
        # Sample new points at which to query for a preference:
        sampled_point_idxs, reward_models = advance(posterior_model, num_samples)

        if ((it + 1) % 5 == 0):
            plot_progress(save_folder, points_to_sample, points_per_dimension, state_dim, posterior_model, it + 1, reward_models, 0) #dimension = 0 for visualization


        # Obtain coordinate points corresponding to these indices, and 
        # store the objective function values:
        sampled_points = np.empty((num_samples, state_dim))
        temp_obj_values = []
        for j in range(num_samples):
           
            sampled_point_idx = sampled_point_idxs[j]
            # Coordinate point representation:
            sampled_points[j, :] = points_to_sample[sampled_point_idx]
            
            sample_idx = it * num_samples + j
            # Objective function value:
            temp_obj_values.append(objective_function(points_to_sample[sampled_point_idx]))

        objective_values.append(temp_obj_values)
        # Obtain a preference between each pair of samples, and store all
        # of the new information.
        for j in range(num_pref):
           
            # Sampled points to compare:
            idx1 = sampled_point_idxs[preference_queries[j][0]]
            idx2 = sampled_point_idxs[preference_queries[j][1]]
            
            # Convert samples to coordinate point representation:
            sampled_pts = np.vstack((points_to_sample[idx1], 
                                    points_to_sample[idx2]))           

            # Query to obtain new preference:
            preference = get_hartmann_preference(points_to_sample[idx1], points_to_sample[idx2], state_dim)    

            # Update the data:            
            data_pts[2*pref_count: 2 * pref_count + 2, :] = sampled_pts
            data_pt_idxs[pref_count, :] = [idx1, idx2]
            labels[pref_count, :] = [1 - preference, preference]      

            pref_count += 1

    # Save the results for this experimental run:
    io.savemat(save_folder + 'Opt_2D_900_' + str(num_samples) + '_samples_' + \
               'vary_obj_run_' + \
        str(run_num) + '.mat', {'data_pts': data_pts, 
           'data_pt_idxs': data_pt_idxs, 'labels': labels, 
            'objective_values': objective_values, 
            'signal_variance': signal_variance, 'lengthscale': lengthscale, 
            'GP_noise_var': GP_noise_var, 'preference_noise': preference_noise})
    
    