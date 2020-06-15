"""
With buffer, use LineCoSpar to optimize in cartpole environment with n = 1.
"""

import numpy as np
import os
import time
from cartpole import *

import scipy.io as io
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from pref_sigmoid import advance, feedback
from utils import *

preference_noise = 0.005 #how noisy are the user's preferences
lower_bound = -5
upper_bound = 5
state_dim = 4
c = 0
discr_param = 1
# Number of times to sample the GP model on each iteration. The algorithm 
# will obtain preferences between all non-overlapping groups of num_samples 
# samples.

# User can't remember too many trials, so we compare the current one to only the past one
num_samples = 1
buffer_size = 1
k = 1

num_trials = 200   # Total number of posterior samples/trials TODO 150
num_iterations = int(np.ceil(num_trials / num_samples))
   
run_nums = np.arange(10)     # Repeat test once for each number in this list.


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

def get_human_pref(state1, state2,first_iter = False):
    #state2 is the new point.
    
    kp=np.array([[state1[0],10*state1[1] + 75],
        [0.,0.]])
    kd=np.array([[state1[2],state1[3]],
                    [0.,0.]])
    # if first_iter:
        # print('hi')
        # r1 = run_sim(kp,kd,True) 
    # else:
    r1 = run_sim(kp,kd)
    kp=np.array([[state2[0],10*state2[1] + 75],
            [0.,0.]])
    kd=np.array([[state2[2],state2[3]],
                [0.,0.]])
    r2 = run_sim(kp,kd,True)

    # print(r1)
    print(r2)

    pref = input("Type 1 if you prefer the previous, 2 if the current, or which one to repeat (r1, r2)")
    
    while pref != '1' and pref != '2' and pref != '0.5':
        if pref == 'r1':
            print("repeating first sample")
            kp=np.array([[state1[0],10*state1[1] + 75],
            [0.,0.]])
            kd=np.array([[state1[2],state1[3]],
                [0.,0.]])
            r1 = run_sim(kp,kd,True)
        elif pref == 'r2':
            print("repeating second sample")
            kp=np.array([[state2[0],10*state2[1] + 75],
            [0.,0.]])
            kd=np.array([[state2[2],state2[3]],
                [0.,0.]])
            r2 = run_sim(kp,kd,True)
        else:
            pref = 0.5
        pref = input("Type 1 if you prefer the previous, 2 if the current.")

    try:
        if pref == '0.5':
            return float(pref), r1, r2
        return int(pref) - 1, r1, r2
    except:
        "input not understood, so returning 0.5"
        return 0.5, r1, r2

# Folder in which to save the results.
save_folder = 'Results/seq_cartpole_%s/' % str(time.time())[-3:]

num_pref = int(buffer_size * (buffer_size - 1) / 2 + buffer_size * \
(num_trials - buffer_size))
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  

# Each iteration of the outer loop is a new repetition of the experiment.
for i, run_num in enumerate(run_nums):
    simulation_data = []
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
    best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, discr_param)

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

        prev_best_point = points_to_sample[posterior_model['best_idx']]
        obj_best =  get_obj_val(prev_best_point)
        obj_bests.append(obj_best)

        # Sample new points at which to query for a preference:
        sampled_point_idxs, reward_models = advance(posterior_model, num_samples)

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
            temp_obj_values.append(get_obj_val(points_to_sample[idx]))
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
            # print(buffer_points)
            if len(buffer_points)==0:
                preference, res1, res2 = get_human_pref(np.asarray(sampled_point), np.asarray(sampled_point),first_iter=True)  

            if len(buffer_points)>=1:
                for j in range(buffer_size):
                    buffer_point_idx = buffer_idxs[j]
                    buffer_point = buffer_points[j]    # Process next point in buffer
                    if np.array_equal(np.asarray(buffer_point),np.asarray(sampled_point)):
                        # print("same")
                        same_count += 1
                        preference = 0.5

                    else:
                        pt1 = np.asarray(buffer_point)
                        pt2 = np.asarray(sampled_point)
                        # preference, res1, res2 = get_control_pref(pt1, pt2) 
                        # #human 
                        if it == 0: 
                            print(it)
                            preference, res1, res2 = get_human_pref(pt1, pt2,first_iter=True)  
                        else:  
                            preference, res1, res2 = get_human_pref(pt1, pt2)  
                        simulation_data.append([pt1, pt2, preference, res1, res2])                

                    # Update the data:            
                    data_pts[2*pref_count: 2 * pref_count + 2, :] = \
                        np.vstack((buffer_point, sampled_point))
                    
                    data_pt_idxs[pref_count, :] = [buffer_point_idx, sampled_point_idx]

                    labels[pref_count, :] = [1 - preference, preference]
                    
                    pref_count += 1
        
            buffer_points = buffer_points[:-1] + [sampled_point]
            buffer_idxs = buffer_idxs[:-1] + [sampled_point_idx]
        
        
        #recalculate prior:

        #get new subspace on every 1 iterations


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

        best_point, GP_prior_cov, points_to_sample = get_prior_randomdir(state_dim, lower_bound, upper_bound, discr_param, visited_point_list, prev_best_point)
            
        GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)  # Inverse of covariance matrix
    # Save the results for this experimental run:
    save_file = save_folder + \
            'run_' + str(run_num) + '.mat'
    np.save('%s.npy' % str(save_folder + 'sim_data_' + str(run_num)),np.array(simulation_data))
    io.savemat(save_file, {'data_pts': data_pts, 
        'data_pt_idxs': data_pt_idxs, 'labels': labels,
        'objective_values': objective_values,'obj_bests': obj_bests})
        # ,'obj_bests': obj_bests
        