"""
Helper functions
"""
import numpy as np
import os
import time
import scipy.io as io
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ortho_group
import scipy
from scipy.optimize import Bounds

def get_random_direction(state_dim):
    """
    creates a random directional vector in state_dim dimensions
    """
    # print(state_dim)
    direction = np.random.normal(size=state_dim)
    # print(direction)
    direction /= np.linalg.norm(direction)
    # direction *= ub - lb  # scale direction with parameter ranges
    return direction

def is_valid_point(pt, lb, ub):
    # print(pt)
    if any(x < lb for x in pt):
        return False
    if any(x > ub for x in pt):
        return False
    return True

def get_prior_randomdir(state_dim, lb, ub, dist, queried_points = [], old_best_point = None, signal_variance = 0.0001):

    points_to_sample = []
    if old_best_point is not None:
        best_point = old_best_point
    else:
        best_point = [np.random.uniform(lb, ub) for i in range(state_dim)]
        points_to_sample.append(best_point)
    # print('i')
    direction = get_random_direction(state_dim)

    #get points along line in negative direction
    n = 1
    while 1:
        new_point = best_point + n*dist*direction
        if is_valid_point(new_point, lb, ub):
            points_to_sample.append(new_point)
        else:
            break
        n += 1

    #get points along line in positive direction
    n = 1
    while 1:
        new_point = best_point - n*dist*direction
        if is_valid_point(new_point, lb, ub):
            points_to_sample.append(new_point)
        else:
            break
        n += 1
    # print("new subspace len")
    # print(len(points_to_sample))
    if len(points_to_sample) == 0:
        return get_prior_randomdir(state_dim, lb, ub, dist, queried_points, old_best_point, signal_variance)
        
    prior_dim = len(points_to_sample)
    # print(prior_dim)
    if np.array(queried_points).size != 0:
        prior_dim += len(queried_points)
        queried_points.extend(points_to_sample)
        points_to_sample = queried_points
    # print(prior_dim)
    lengthscales = [0.15] * state_dim          # Larger values = smoother reward function
    GP_prior_cov = signal_variance * np.ones((prior_dim, prior_dim)) 
    GP_noise_var = 1e-5        # GP model noise--need at least a very small
                                    # number to ensure that the covariance matrix
                                    #  is invertible.

    for i in range(prior_dim):

        pt1 = points_to_sample[i]
        
        for j in range(prior_dim):
            
            pt2 = points_to_sample[j]
            
            for dim in range(state_dim):
                
                lengthscale = lengthscales[dim]
                
                if lengthscale > 0:
                    GP_prior_cov[i, j] *= np.exp(-0.5 * ((pt2[dim] - pt1[dim]) / lengthscale)**2) 

                elif lengthscale == 0 and pt1[dim] != pt2[dim]:
                    
                    GP_prior_cov[i, j] = 0
    
    GP_prior_cov += GP_noise_var * np.eye(prior_dim)
    
    return best_point, GP_prior_cov, points_to_sample

def plot_progress_2(coefs,num_iter, save_folder, points_to_sample,state_dim, posterior_model, pref_num, reward_models,
num_samples = 3, lower_bound = 0, upper_bound = 1):
    # plt.rcParams["font.family"] = "Arial"
    x, min_val = get_min(coefs,state_dim)
    x, max_val = get_max(coefs,state_dim)
    fn_range = max_val - min_val

    # print("min and max")
    # print(min_val)
    # print(max_val)
    # Unpack model posterior:
    post_mean = posterior_model['mean']
    cov_evecs = np.real(posterior_model['cov_evecs'])
    cov_evals = posterior_model['cov_evals']
    
    # Construct posterior covariance matrix:
    post_cov = cov_evecs @ np.diag(cov_evals) @ np.linalg.inv(cov_evecs)
    
    # Posterior standard deviation at each point:
    post_stdev = np.sqrt(np.diag(post_cov))
    
    # rcParams.update({'font.size': 40})

    fig, axs = plt.subplots(2,figsize = (9, 18))
    
    x_values = np.arange(len(points_to_sample)) #what about visited points?

    # Plot posterior mean and standard deviation:
    axs[0].plot(x_values, post_mean, color = 'blue', linewidth = 5)
    axs[0].fill_between(x_values, post_mean - 2*post_stdev, 
                    post_mean + 2*post_stdev, alpha = 0.3, color = 'blue')
    

    # Plot posterior samples:
    for j in range(num_samples):
        
        reward_model = reward_models[j]
        
        axs[0].plot(x_values, reward_model, color = 'green',
                linestyle = '--', linewidth = 3)
    axs[0].set_yticks([-0.02, 0, 0.02, 0.04,0.06])
    # print('fn rng')
    # print(fn_range)
    # print('min')
    # print(min_val)
    # print('max')
    # print(max_val)
    # for pt in points_to_sample[:2]:
    #     print('val')
    #     print(poly_obj_fn(pt, coefs))
        # print("final")
        # print((poly_obj_fn(pt, coefs)-min_val))

    actuals = [(poly_obj_fn(point, coefs)-min_val)/(fn_range) for point in points_to_sample]
    # print(actuals)
    
    axs[1].plot(x_values, actuals, color = 'red',linewidth = 5)
    # print(np.min(actuals))
    axs[0].set_xlabel('Visited actions')
    axs[1].set_xlabel('Visited actions')
    # axs[0].set_ylim(-0.03, 0.07)
    # axs[1].set_ylim(0, 3.3)
    # if any(actuals) < 0 or any(actuals)  > 1:
    #     print('bug')
    #     sys.exit(0)

    if num_iter == 0:
        axs[0].legend(['Posterior','Posterior samples'], loc = 'upper right',
            fontsize = 40)
    
    # plt.xticks(np.li
    save_file = save_folder + 'test_dim_' + str(state_dim) + '_' +str(pref_num) + \
    '_preferences_' + str(time.time())[-3:] + '.mat'
    io.savemat(save_file, {'x_values':x_values, 'pmean':post_mean, 'reward_models':reward_models, 'pstd':post_stdev, 'actuals':actuals})

    plt.savefig(save_folder + 'test_dim_' + str(state_dim) + '_' +str(pref_num) + \
    '_preferences_' + str(time.time())[-3:] + '.png')
    plt.close('all')
    return actuals - post_mean


# from hpolib.benchmarks.synthetic_functions import Hartmann6 #pip install git+https://github.com/automl/HPOlib2

def cartesian(arrays, out=None):
        """
        Generate a cartesian product of input arrays.

        Paxrameters
        ----------
        arrays : list of array-like
                1-D arrays to form the cartesian product of.
        out : ndarray
                Array to place the cartesian product in.

        Returns
        -------
        out : ndarray
                2-D array of shape (M, len(arrays)) containing cartesian products
                formed of input arrays.

        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
                     [1, 4, 7],
                     [1, 5, 6],
                     [1, 5, 7],
                     [2, 4, 6],
                     [2, 4, 7],
                     [2, 5, 6],
                     [2, 5, 7],
                     [3, 4, 6],
                     [3, 4, 7],
                     [3, 5, 6],
                     [3, 5, 7]])

        """

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n / arrays[0].size
        m = int(m)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
        return out

def camel_objective_function(pt):
    """converted from https://www.sfu.ca/~ssurjano/camel3.html
       bounds [-5, 5]

    """
    x1 = pt[0]
    x2 = pt[1]
    term1 = 2*x1*x1
    term2 = -1.05*(x1**4)
    term3 = (x1**6) / 6
    term4 = x1*x2
    term5 = x2**2
    y = term1 + term2 + term3 + term4 + term5
    return -y

def hartmann_objective_function(pt, dim):
    """3d or 6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
    """
    alpha = [1.00, 1.20, 3.00, 3.20]
    if dim == 3:
        A = np.array([[3.0, 10.0, 30.0],
                        [0.1, 10.0, 35.0],
                        [3.0, 10.0, 30.0],
                        [0.1, 10.0, 35.0]])
        P = 0.0001 * np.array([[3689, 1170, 2673],
                                [4699, 4387, 7470],
                                [1090, 8732, 5547],
                                [381, 5743, 8828]])
    elif dim == 6:
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])
    else:
        raise ValueError('Hartmann function should have either 3 or 6 dimensions')

    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(dim):
            internal_sum += A[i, j] * (pt[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)

    return external_sum

def hartmann_grad(pt, dim):
    alpha = [1.00, 1.20, 3.00, 3.20]
    if dim == 6:
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])
    else:
        raise ValueError('must be 6 dimensions')

    external_sum = np.zeros((6,1))
    for i in range(4):
        internal_sum = 0
        internal_prime = 0

        for j in range(dim):
            internal_sum += A[i, j] * (pt[j] - P[i, j]) ** 2

        # for k in range(dim):
            internal_prime += 2 * A[i, j] * (pt[j] - P[i, j])
        
            external_sum[i] += alpha[i] * (-internal_prime) * np.exp(-internal_sum)
    # print(external_sum)
    return external_sum.reshape(6,)


def get_points_with_values(state_dim = 3, points_per_dimension = 10):
    #for plotting purposes
    lower_bound = 0
    upper_bound = 1
    space_range = np.linspace(lower_bound, upper_bound, points_per_dimension)

    if state_dim == 3:
        all_points = cartesian((space_range, space_range, space_range))
    else:
        all_points = cartesian((space_range, space_range, space_range, space_range, space_range, space_range))
    obj_vals = [hartmann_objective_function(point, state_dim) for point in all_points]
    sorted_vals = np.sort(obj_vals)
    # print(obj_vals[-1])
    sorted_vals_dict = dict(zip(sorted_vals, np.arange(len(obj_vals))))
    np.save("sorted_points_%dds_%d.npy" % (state_dim, points_per_dimension), sorted_vals_dict)

def get_labeled_points_with_values(state_dim = 6, points_per_dimension = 10):
    lower_bound = 0
    upper_bound = 1
    space_range = np.linspace(lower_bound, upper_bound, points_per_dimension)

    if state_dim == 3:
        all_points = cartesian((space_range, space_range, space_range))
    else:
        all_points = cartesian((space_range, space_range, space_range, space_range, space_range, space_range))
    obj_vals = [[tuple(point), hartmann_objective_function(point, state_dim)] for point in all_points]

    sorted_vals = sorted(obj_vals, key = lambda x: x[1])
    # print(obj_vals[-1])
    just_obj_vals = [v[1] for v in sorted_vals]
    values_to_put = [(i, v[0]) for i, v in enumerate(sorted_vals)]
    sorted_vals_dict = dict(zip(just_obj_vals, values_to_put))
    # print(sorted_vals_dict)
    np.save("sorted_labeled_points_%dds_%d.npy" % (state_dim, points_per_dimension), sorted_vals_dict)

def get_camel_preference(pt1,pt2):
    obj1 = camel_objective_function(pt1)
    obj2 = camel_objective_function(pt2)
    
    if obj2 > obj1:
        return 1
    elif obj1 > obj2:
        return 0
    else:
        return np.random.choice(2)

def get_hartmann_preference(pt1, pt2, state_dim):
    obj1 = hartmann_objective_function(pt1,state_dim)
    obj2 = hartmann_objective_function(pt2, state_dim)
    
    if obj2 > obj1:
        return 1
    elif obj1 > obj2:
        return 0
    else:
        return np.random.choice(2)


def plot_progress_4d(save_folder, points_to_sample, num_samples, posterior_model, pref_num, reward_models, dim_to_keep):
    # Unpack model posterior:
    post_mean = posterior_model['mean']
    cov_evecs = np.real(posterior_model['cov_evecs'])
    cov_evals = posterior_model['cov_evals']
    
    # Construct posterior covariance matrix:
    post_cov = cov_evecs @ np.diag(cov_evals) @ np.linalg.inv(cov_evecs)
    
    # Posterior standard deviation at each point:
    post_stdev = np.sqrt(np.diag(post_cov))
    
    plt.figure(figsize = (8, 6))
    
    # x_values = [point[dim_to_keep] for point in points_to_sample] #what about visited points?


    x = [point[0] for point in points_to_sample]
    y = [point[1] for point in points_to_sample]
    z = [point[2] for point in points_to_sample]
    fig = plt.figure()

    # plt.fill_between(x_values, post_mean - 2*post_stdev, 
    #                  post_mean + 2*post_stdev, alpha = 0.3, color = 'blue')
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=post_mean, cmap=plt.hot())
    # Plot posterior samples:
    for j in range(num_samples):
        print(len(reward_model))
        reward_model = reward_models[j]
        
        ax.scatter(x, y, z, c=reward_model, cmap=plt.hot(), linestyle = '--',)
    
    plt.tight_layout()
    
    fig.colorbar(img)

    plt.savefig(save_folder + '4d_test_' + str(pref_num) + \
            '_preferences_' + str(time.time())[-3:] + '.png')




def get_prior(state_dim, space_range, dim_to_keep, queried_points = [], old_best_point = None, signal_variance = 0.0001):
    
    if old_best_point:
        best_point = old_best_point
    else:
        best_point = [np.random.choice(space_range) for i in range(state_dim)]

    points_per_dimension = space_range.shape[0]

    points_to_sample = [list(best_point) for _ in range(points_per_dimension)]

    for i in range(len(space_range)):
        points_to_sample[i][dim_to_keep] = space_range[i]

    prior_dim = points_per_dimension
    
    if np.array(queried_points).size != 0:
        prior_dim += len(queried_points)
        queried_points.extend(points_to_sample)
        points_to_sample = queried_points

    lengthscales = [0.15] * state_dim          # Larger values = smoother reward function
    GP_prior_cov = signal_variance * np.ones((prior_dim, prior_dim)) 
    GP_noise_var = 1e-5        # GP model noise--need at least a very small
                                    # number to ensure that the covariance matrix
                                    #  is invertible.

    for i in range(prior_dim):

        pt1 = points_to_sample[i]
        
        for j in range(prior_dim):
            
            pt2 = points_to_sample[j]
            
            for dim in range(state_dim):
                
                lengthscale = lengthscales[dim]
                
                if lengthscale > 0:
                    GP_prior_cov[i, j] *= np.exp(-0.5 * ((pt2[dim] - pt1[dim]) / lengthscale)**2) 

                elif lengthscale == 0 and pt1[dim] != pt2[dim]:
                    
                    GP_prior_cov[i, j] = 0
    
    GP_prior_cov += GP_noise_var * np.eye(prior_dim)
    
    return best_point, GP_prior_cov, points_to_sample

def get_max(coefs,state_dim):
    b = Bounds([0]*state_dim, [1]*state_dim)
    res = scipy.optimize.minimize(neg_poly_obj_fn,0.5*np.ones((state_dim,1)),args=coefs,bounds=b)
    return res.x,-res.fun

def get_min(coefs,state_dim):
    b = Bounds([0]*state_dim, [1]*state_dim)
    res = scipy.optimize.minimize(poly_obj_fn,0.5*np.ones((state_dim,1)),args=coefs,bounds=b)
    return res.x,res.fun

def neg_poly_obj_fn(pt,coefs):
    pt = np.array(pt).reshape((len(coefs),))
    coefs1 = coefs[:,0]
    coefs2 = coefs[:,1]
    vals = np.polynomial.polynomial.polyval(pt,coefs1)
    return -np.dot(vals,coefs2)

def poly_obj_fn(pt,coefs):
    pt = np.array(pt).reshape((len(coefs),))
    coefs1 = coefs[:,0]
    coefs2 = coefs[:,1]
    vals = np.polynomial.polynomial.polyval(pt,coefs1)
    return np.dot(vals,coefs2)
    
def get_poly_preference(pt1, pt2,coefs):
    obj1 = poly_obj_fn(pt1,coefs)
    obj2 = poly_obj_fn(pt2, coefs)
    
    if obj2 > obj1:
        return 1
    elif obj1 > obj2:
        return 0
    else:
        return np.random.choice(2)

