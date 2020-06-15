"""
Use LineCoSpar to optimize over 3D/6D hartmann objective function for n = 1.
"""

import numpy as np
import os
import time
import scipy.io as io
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from pref_sigmoid import advance, feedback
from utils import *

preference_noise = 0.005 #how noisy are the user's preferences
lower_bound = 0
sub_rs = [0]
upper_bound = 1
state_dim = 3
c = 0
min_int = 0

# Number of times to sample the GP model on each iteration. The algorithm 
# will obtain preferences between all non-overlapping groups of num_samples 
# samples.

# User can't remember too many trials, so we compare the current one to only the past one
num_samples = 1
buffer_size = 1
k = 1

num_trials = 300   # Total number of posterior samples/trials TODO 150
num_iterations = int(np.ceil(num_trials / num_samples))
   
run_nums = np.arange(10)     # Repeat test once for each number in this list.

            

# Folder in which to save the results.
save_folder = 'Results/GP_LineCoSpar/LineCoSpar_hartmann_1update_3subspace_buffer_%s/' % str(time.time())[-3:]

num_pref = int(buffer_size * (buffer_size - 1) / 2 + buffer_size * \
(num_trials - buffer_size))
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):
    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * num_iterations * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref * num_iterations, 2)).astype(int)
    labels = np.empty((num_pref * num_iterations, 2))
    
    buffer_points = []
    buffer_idxs = []
    obj_bests = []
    objective_values = [] #for plotting purposes only
    same_count = 0
    #Get first prior
    visited_points = {} #dictionary of visited (sampled) points
    objective_values = [] #for plotting purposes only
    best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, 0.1)

    GP_prior_cov_inv =  np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix
    prev_best_point = best_point
    pref_count = 0

    # In each iteration, we sample num_samples points from the model posterior  
    for it in range(num_iterations):

        # obtain a posterior to pass to the advance function
        if it % 50 == 0:
            print('Run %i of %i, iteration %i of %i' % (i + 1, len(run_nums), 
            it + 1, num_iterations))
            print("Num same: " + str(same_count))
        
        # Preference data observed so far (used to train GP preference model):
        X = data_pt_idxs[: pref_count, :]
        y = labels[: pref_count, 1]
        
        # Update the Gaussian process preference model:
        posterior_model = feedback(X, y, GP_prior_cov_inv, preference_noise)

        random_choice = np.random.rand()
        prev_best_point = points_to_sample[posterior_model['best_idx']]
        obj_best =  hartmann_objective_function(prev_best_point,state_dim)
        obj_bests.append(obj_best)

        # Sample new points at which to query for a preference:
        sampled_point_idxs, reward_models = advance(posterior_model, num_samples)

        if ((it + 1) % 50== 0):
            plot_progress_2(it, save_folder, points_to_sample,state_dim, posterior_model, it + 1, reward_models,num_samples)

        # Obtain coordinate points corresponding to these indices
        sampled_points = np.empty((num_samples, state_dim)) 
        for j in range(num_samples):
            sampled_point_idx = sampled_point_idxs[j]
            sampled_points[j, :] = points_to_sample[sampled_point_idx]

        prev_num_visited_points = len(visited_points.keys())
        count = len(visited_points.keys())
        
        # Get objective values for each point, and add them to our list
        # of points if we have not visited them yet.
        temp_obj_values = []
        for idx in sampled_point_idxs:
            temp_obj_values.append(hartmann_objective_function(points_to_sample[idx],state_dim))
            if idx >= prev_num_visited_points: #it's a point in the new subspace
                tuple_point = tuple(points_to_sample[idx])
                if tuple_point not in visited_points: #if it hasn't been visited before, add it
                    visited_points[tuple_point] = count
                    #the global index holds true throughout different subspaces
                    count += 1

        objective_values.append(temp_obj_values)
        
        for samplind in range(len(sampled_point_idxs)):
            # Obtain a preference between the newly-sampled point and each
            # sample in the buffer:
            sampled_point_idx = sampled_point_idxs[samplind]
            sampled_point = points_to_sample[sampled_point_idx]
            # print(sampled_point_idx)
            #If it's a point in the new subspace, we want to save its global index
            # if sampled_point_idx >= prev_num_visited_points:
                # sampled_point_idx = visited_points[tuple(sampled_point)] 

            if len(buffer_points)>=1:
                for j in range(buffer_size):
                    buffer_point_idx = buffer_idxs[j]
                    buffer_point = buffer_points[j]    # Process next point in buffer
                    # print("buffer")
                    # print(buffer_point)
                    # print(type(buffer_point))
                    # print("\n sampled")
                    # print(sampled_point)
                    # print(type(sampled_point))
                    if np.array_equal(np.asarray(buffer_point),np.asarray(sampled_point)):
                        # print("same")
                        same_count += 1
                    # else:
                    #     print("different")
                    # Query to obtain new preference:
                    preference = get_hartmann_preference(buffer_point, sampled_point, state_dim)                    

                    # Update the data:            
                    data_pts[2*pref_count: 2 * pref_count + 2, :] = \
                        np.vstack((buffer_point, sampled_point))
                    
                    data_pt_idxs[pref_count, :] = [buffer_point_idx, sampled_point_idx]

                    labels[pref_count, :] = [1 - preference, preference]
                    
                    pref_count += 1
        
            buffer_points = buffer_points[:-1] + [sampled_point]
            buffer_idxs = buffer_idxs[:-1] + [sampled_point_idx]
        
        
        #recalculate prior:

        #get new subspace on every 3 iterations

        if it % 3 == 2:

            for spot, idx in enumerate(buffer_idxs):
                if idx > prev_num_visited_points: #idx1
                    pt0 = points_to_sample[idx]
                    buffer_idxs[spot] = visited_points[tuple(pt0)]
            for row in data_pt_idxs:
                if row[0] > prev_num_visited_points: #idx1
                    pt1 = points_to_sample[row[0]]
                    row[0] = visited_points[tuple(pt1)]
                if row[1] > prev_num_visited_points: #idx2
                    pt2 = points_to_sample[row[1]]
                    row[1] = visited_points[tuple(pt2)]

            visited_point_list = [list(tup) for tup in visited_points.keys()]

            best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, 0.1, visited_point_list, prev_best_point)
                
            GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix
    # Save the results for this experimental run:
    save_file = save_folder + 'Opt_' + str(state_dim) + 'D_LineCoSpar_' + str(num_samples) + '_samples_' + \
            'vary_obj_run_' + str(run_num) + '.mat'

    io.savemat(save_file, {'data_pts': data_pts, 
        'data_pt_idxs': data_pt_idxs, 'labels': labels,
        'objective_values': objective_values,'obj_bests': obj_bests})
    
    