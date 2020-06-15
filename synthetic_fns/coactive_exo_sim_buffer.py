"""
TODO"""

import numpy as np
import os
import time
import scipy.io as io
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from opt_utils_exo import *
from utils import hartmann_objective_function, get_hartmann_preference,plot_progress_2
from scipy.optimize import approx_fprime


preference_noise = 0.005 #how noisy are the user's preferences
sub_rs = [0]
state_dim = 6
discretization = [0.0071,0.0333,0.01,0.0025,1,1]
lbs = [0.08,0.85,0.25,0.065,5,10]
ubs = [0.18,1.15,0.3,0.075,10,15]
c = 0
min_int = 0

coactive_increase = 0.1 #amount  to increment... TODO proportional to the gradient?


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
save_folder = 'Results/GP_LineCoSpar/exobounds_coactive_buffer_%s/' % str(time.time())[-3:]

num_pref = int(buffer_size * (buffer_size - 1) / 2 + buffer_size * \
(num_trials - buffer_size))
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):
    # Initialize data matrix and label vector:
    data_pts = []
    data_pt_idxs = []
    neg_points = 0
    labels = []

    buffer_points = []
    buffer_idxs = []
    objective_values = [] #for plotting purposes only
    same_count = 0
    #Get first prior
    visited_points = {} #dictionary of visited (sampled) points
    obj_bests = []
    objective_values = [] #for plotting purposes only
    prev_best_point = [np.random.uniform(lbs[i], ubs[i]) for i in range(state_dim)]
    
    best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lbs, ubs, discretization, visited_points, prev_best_point)

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
        X = np.array(data_pt_idxs)
        if labels == []:
            y = np.empty((1,1))
        else:
            y = np.array(labels)[:,1]
        
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
                # print("adding sampled action")
                # print(tuple_point)
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

            if len(buffer_points)>=1:
                for j in range(buffer_size):
                    buffer_point_idx = buffer_idxs[j]
                    buffer_point = buffer_points[j]    # Process next point in buffer
                    if np.array_equal(np.asarray(buffer_point),np.asarray(sampled_point)):
                        # print("same")
                        same_count += 1
                    # else:
                    #     print("different")
                    # Query to obtain new preference:
                    preference = get_hartmann_preference(buffer_point, sampled_point, state_dim)                    


                    data_pts.append(np.vstack((buffer_point, sampled_point)))
                    # print("adding pref")
                    data_pt_idxs.append([buffer_point_idx, sampled_point_idx])
                    labels.append([1 - preference, preference])     
                    

                    if preference == 0:
                        curr_idx = buffer_point_idx
                        curr_pt = buffer_point
                    else:
                        curr_idx = sampled_point_idx
                        curr_pt = sampled_point
                    # curr_grad = hartmann_grad(curr_pt, state_dim)
                    eps = np.sqrt(np.finfo(float).eps)
                    args = (6,) 

                    curr_grad = approx_fprime(curr_pt,hartmann_objective_function,eps,*args)
                    
                    max_dir = np.argmax(abs(curr_grad))
                    if abs(curr_grad[max_dir]) > 1:
                        coactive_pt = np.copy(np.array(curr_pt))
                        coactive_pt[max_dir] = curr_pt[max_dir] + (coactive_increase * np.sign(curr_grad[max_dir])) 

                        

                        improve = hartmann_objective_function(coactive_pt,state_dim) - hartmann_objective_function(curr_pt,state_dim)
                        if improve > 0:
                            #add coactive pt to visited points
                            tuple_point = tuple(coactive_pt)
                            if tuple_point not in visited_points: #if it hasn't been visited before, add it
                                visited_points[tuple_point] = count
                                coactive_idx = len(points_to_sample)
                                points_to_sample.append(coactive_pt)
                                count += 1 
                            else:
                                coactive_idx = visited_points[tuple_point]
                            coactive_pts = np.vstack((curr_pt, coactive_pt)) 
                            data_pts.append(coactive_pts)
                            data_pt_idxs.append([curr_idx, coactive_idx])
                            labels.append([0, 1])   #prefer second point
                        else: 
                            neg_points += 1

                    pref_count += 1
        
            buffer_points = buffer_points[:-1] + [sampled_point]
            buffer_idxs = buffer_idxs[:-1] + [sampled_point_idx]
        
        
        #recalculate prior:

        # get new subspace on every 3 iterations
        # if it % 3 == 2:
        
        for spot, idx in enumerate(buffer_idxs):
            # if idx > prev_num_visited_points: #idx1
            pt0 = points_to_sample[idx]
            buffer_idxs[spot] = visited_points[tuple(pt0)]


        for row in data_pt_idxs:
            if row[0] > prev_num_visited_points: #idx1
                pt1 = points_to_sample[row[0]]
                row[0] = visited_points[tuple(pt1)]
            if row[1] > prev_num_visited_points: #idx2
                # print(row[1])
                # print(visited_points)
                pt2 = points_to_sample[row[1]]
                row[1] = visited_points[tuple(pt2)]

        visited_point_list = [list(tup) for tup in visited_points.keys()]
        
        best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lbs, ubs, discretization, visited_point_list, prev_best_point)
                
        GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix
    # Save the results for this experimental run:
    save_file = save_folder + 'Opt_' + str(state_dim) + 'D_LineCoSpar_' + str(num_samples) + '_samples_' + \
            'vary_obj_run_' + str(run_num) + '.mat'

    io.savemat(save_file, {'data_pts': data_pts, 
        'data_pt_idxs': data_pt_idxs, 'labels': labels,
        'objective_values': objective_values,'obj_bests': obj_bests})
    
    