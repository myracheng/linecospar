"""
Use LineCoSpar to optimize over 3D/6D hartmann objective function.
"""

import numpy as np
import os
import time
import scipy.io as io
import itertools
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
rcParams.update({'font.size': 24})
from pref_sigmoid import advance, feedback
from utils import *

lower_bound = 0
upper_bound = 1
state_dim = 6
preference_noise = 0.005
num_samples = 3

num_trials = 160*3   # Total number of posterior samples/trials TODO 150
num_iterations = int(np.ceil(num_trials / num_samples))

run_nums = np.arange(10)     # Repeat test once for each number in this list.


# Folder in which to save the results.
save_folder = 'Results/randdir_hartmann_%s/' % (str(time.time())[-3:])
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  
    
preference_queries = list(itertools.combinations(np.arange(num_samples), 2))
num_pref = len(preference_queries)    # Pairwise preferences per iteration

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):

    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * num_iterations * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref * num_iterations, 2)).astype(int)
    labels = np.empty((num_pref * num_iterations, 2))

    visited_points = {} #dictionary of visited (sampled) points
    objective_values = [] #for plotting purposes only
    obj_bests = []
    #Get first prior
    best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, 0.1)

    GP_prior_cov_inv =  np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix
    prev_best_point = best_point
    pref_count = 0

    # In each iteration, we sample num_samples points from the model posterior, 
    # and obtain all possible pairwise preferences between these points.   
    for it in range(num_iterations):

        # obtain a posterior to pass to the advance function
        if it % 50 == 0:
            print('Run %i of %i, iteration %i of %i' % (i + 1, len(run_nums), 
            it + 1, num_iterations))
        
        # Preference data observed so far (used to train GP preference model):
        X = data_pt_idxs[: pref_count, :]
        y = labels[: pref_count, 1]
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)

        prev_best_point = points_to_sample[posterior_model['best_idx']]

        obj_best =  hartmann_objective_function(prev_best_point,state_dim)
        obj_bests.append(obj_best)

        # Sample new points at which to query for a preference:
        sampled_point_idxs, reward_models = advance(posterior_model, num_samples)

        # if ((it) % 50 == 0):
        #     plot_progress_2(it, save_folder, points_to_sample,state_dim, posterior_model, it + 1, reward_models)

        # Obtain coordinate points corresponding to these indices
        sampled_points = np.empty((num_samples, state_dim)) 
        for j in range(num_samples):
            sampled_point_idx = sampled_point_idxs[j]
            sampled_points[j, :] = points_to_sample[sampled_point_idx]

        prev_num_visited_points = len(visited_points.keys())
        count = prev_num_visited_points
        
        # Get objective values for each point for plotting purposes
        # Add sampled points to our list of visited points.
        temp_obj_values = []
        for idx in sampled_point_idxs:
            temp_obj_values.append(hartmann_objective_function(points_to_sample[idx],state_dim))
            if idx >= prev_num_visited_points: #it's a point in the new subspace
                tuple_point = tuple(points_to_sample[idx])
                if tuple_point not in visited_points: #if it hasn't been visited before, add it
                    visited_points[tuple_point] = count

                    #the global index holds true throughout the 
                    # iterations, since the first part of the list 
                    # of points, the visited points, don't change.
                    count += 1

        objective_values.append(temp_obj_values)
        
    
        # Obtain a preference between each pair of samples, and store all
        # of the new information.
        for j in range(num_pref):
        
            # Sampled points to compare:
            idx1 = sampled_point_idxs[preference_queries[j][0]]
            idx2 = sampled_point_idxs[preference_queries[j][1]]
            pt1 = points_to_sample[idx1]
            pt2 = points_to_sample[idx2]

            # Convert samples to coordinate point representation:
            sampled_pts = np.vstack((pt1, 
                                    pt2)) 

            # Query to obtain new preference:
            preference = get_hartmann_preference_noisy(pt1, pt2, state_dim,c)     
            # print(preference)
            # Update the data:            
            if idx1 >= prev_num_visited_points:
                idx1 = visited_points[tuple(pt1)] 
            if idx2 >= prev_num_visited_points:
                idx2 = visited_points[tuple(pt2)] 
            data_pts[2*pref_count: 2 * pref_count + 2, :] = sampled_pts
            data_pt_idxs[pref_count, :] = [idx1, idx2]
            labels[pref_count, :] = [1 - preference, preference]      

            pref_count += 1
        
        #Recalculate prior to include the points sampled in this iteration.
        visited_point_list = [list(tup) for tup in visited_points.keys()]
        best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, 0.1, visited_point_list, prev_best_point)
        GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix
    

    # Save the results for this experimental run:
    save_file = save_folder + 'Opt_' + str(state_dim) + 'D_LineCoSpar_' + str(num_samples) + '_samples_' + \
            'vary_obj_run_' + str(run_num) + '.mat'

    io.savemat(save_file, {'data_pts': data_pts, 
        'data_pt_idxs': data_pt_idxs, 'labels': labels,
        'objective_values': objective_values,'obj_bests': obj_bests})
        
        