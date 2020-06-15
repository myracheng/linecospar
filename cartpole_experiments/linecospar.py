"""
Use LineCoSpar to optimize over 4-parameter cartpole space with human input.
"""
import numpy as np
import os
from cartpole import *
import time
import scipy.io as io
import itertools
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from pref_sigmoid import advance, feedback
from utils import *


subject_name = ''

lower_bound = -5
upper_bound = 5
discr_param = 1
state_dim = 4
preference_noise = 0.005
num_samples = 2

num_trials = 100*2  # Total number of posterior samples
num_iterations = int(np.ceil(num_trials / num_samples))

run_nums = np.arange(10)     # Repeat test once for each number in this list.

# Folder in which to save the results.
timestr = str(time.time())[-3:]
# timestr = 'test'
save_folder = 'Results/testpd_%s%s/' % (subject_name, timestr)
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  
    
preference_queries = list(itertools.combinations(np.arange(num_samples), 2))
num_pref = len(preference_queries)    # Pairwise preferences per iteration

def get_obj_val(state1):
    kp1 = 10*state1[1] + 75
    kp=np.array([[state1[0],kp1],
            [0.,0.]])
    kd=np.array([[state1[2],state1[3]],
                [0.,0.]])
    return run_sim(kp,kd) 

def get_control_pref(state1,state2):
    steps1 = get_obj_val(state1)
    steps2 = get_obj_val(state2)
    if steps2 > steps1:
        return 1, steps1, steps2
    elif steps2 == steps1:
        return 0.5, steps1, steps2
    else:
        return 0, steps1, steps2

def get_human_pref(state1, state2):
    kp=np.array([[state1[0],10*state1[1] + 75],
            [0.,0.]])
    kd=np.array([[state1[2],state1[3]],
                [0.,0.]])
    r1 = run_sim(kp,kd,True) 
    kp=np.array([[state2[0],10*state2[1] + 75],
            [0.,0.]])
    kd=np.array([[state2[2],state2[3]],
                [0.,0.]])
    r2 = run_sim(kp,kd,True)
    print(r1)
    print(r2)

    pref = input("0 if you prefer the first, 1 if the second, 0.5 if same")
    try:
        if pref == '0.5':
            return float(pref), r1, r2
        return int(pref), r1, r2
    except:
        "input not understood, so returning 0.5"
        return 0.5, r1, r2

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):
    simulation_data = []
    # Initialize data matrix and label vector:
    data_pts = np.empty((num_pref * num_iterations * 2, state_dim))
    data_pt_idxs = np.zeros((num_pref * num_iterations, 2)).astype(int)
    labels = np.empty((num_pref * num_iterations, 2))

    visited_points = {} #dictionary of visited (sampled) points
    objective_values = [] #for plotting purposes only
    obj_bests = []
    #Get first prior
    best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, discr_param)

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

        obj_best =  get_obj_val(prev_best_point)
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
            temp_obj_values.append(get_obj_val(points_to_sample[idx]))
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
            if (np.array(pt1) == np.array(pt2)).all():
                preference = 0.5
            else:
                # preference, res1, res2 = get_control_pref(pt1, pt2)  
                preference, res1, res2 = get_human_pref(pt1, pt2)     
                simulation_data.append([pt1, pt2, preference, res1, res2])
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
        best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, discr_param, visited_point_list, prev_best_point)
        GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix
    

    # Save the results for this experimental run:
    save_file = save_folder + \
            'run_' + str(run_num) + '.mat'
    np.save('%s.npy' % str(save_folder + 'sim_data_' + str(run_num)),np.array(simulation_data))
    io.savemat(save_file, {'data_pts': data_pts, 
        'data_pt_idxs': data_pt_idxs, 'labels': labels,
        'objective_values': objective_values,'obj_bests': obj_bests})
        
            