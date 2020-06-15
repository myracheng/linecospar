#todo: get all mat files from the folder, add to an array to plot

# -*- coding: utf-8 -*-
"""
Plot results from simulations optimizing 2D randomly-generated synthetic 
objective functions.
"""
import glob, os
from scipy import stats
import numpy as np
import scipy.io as io
from sklearn.preprocessing import normalize
from scipy.optimize import Bounds
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib import rcParams
from matplotlib import rc
import matplotlib as mpl
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots 
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 60,
    "font.size": 75,
    "legend.fontsize": 30,               # Make the legend/label fonts 
    "xtick.labelsize": 50,               # a little smaller
    "ytick.labelsize": 50,
    # "figure.figsize": [your-width,your-height],     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",    # use utf8 input and T1 fonts 
        r"\usepackage[T1]{fontenc}",        # plots will be generated 
        r"\usepackage{amsmath}",
        r"\usepackage{bm}"
        ]                                   # using this preamble
    }
mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)
fig = plt.figure(figsize = (18, 9))
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def get_max(coefs,b):
    res = scipy.optimize.minimize(neg_poly_obj_fn,0.3*np.ones((6,1)),args=coefs,bounds=b)
    return res.x,-res.fun

def get_min(coefs,b):
    res = scipy.optimize.minimize(poly_obj_fn,0.3*np.ones((6,1)),args=coefs,bounds=b)
    return res.x,res.fun

def neg_poly_obj_fn(pt,coefs):
    pt = np.array(pt).reshape((6,))
    coefs1 = coefs[:,0]
    coefs2 = coefs[:,1]
    vals = np.polynomial.polynomial.polyval(pt,coefs1)
    return -np.dot(vals,coefs2)

def poly_obj_fn(pt,coefs):
    pt = np.array(pt).reshape((6,))
    coefs1 = coefs[:,0]
    coefs2 = coefs[:,1]
    vals = np.polynomial.polynomial.polyval(pt,coefs1)
    return np.dot(vals,coefs2)
# Color-blind friendly palette: https://gist.github.com/thriveth/8560036
CB_colors = ['#377eb8', '#4daf4a', '#ff7f00', 
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']                  

colors = CB_colors
   
fig_num = 3

num_runs = 1     # Times experiment was repeated
num_trials = 150    # Total number of posterior samples/trials


alpha = 0.4


b = Bounds([0.08,0.85,0.25,0.065,5.5,10.5],[0.18,1.15,0.3,0.075,9.5,14.5])
dirs = ["../coacties/poly_n1_b1_995_exodisc",
"../coact_poly_n1_b1_0.1_361"]
plot_leg = ['no coactive feedback','with coactive feedback']

for i, mydir in enumerate(dirs):
    os.chdir(mydir)
    obj_vals = np.empty((num_trials,0))
    for filename in glob.glob("*.mat"):
        results = io.loadmat(filename)
        obj = np.array(results['objective_values'])[:num_trials, :]
        coefs = results['coefs']
        x, min_val = get_min(coefs,b)
        x, max_val = get_max(coefs,b)
        fn_range = max_val - min_val
        obj = np.divide(np.subtract(obj,min_val),fn_range)
        obj_vals = np.concatenate((obj_vals, obj), axis=1)

    
    mean =  np.array([np.mean(vals) for vals in obj_vals])[:num_trials]
    stdev = np.array([np.std(vals) for vals in obj_vals])[:num_trials]
    mins = np.array([np.min(vals) for vals in obj_vals])[:num_trials]
    maxs = np.array([np.max(vals) for vals in obj_vals])[:num_trials]
    
    color = colors[i]
    mean_linewidth = 5
    plot_SD = True
    np.save('means_%d.npy'%i,mean)
    np.save('stdev_%d.npy'%i,stdev)
    # # Plot the mean over the trials:    
    plt.plot(np.arange(1, num_trials + 1), mean, color = color,linewidth = mean_linewidth)
    
    mean_err = []
    st_dev = []
    for m in np.arange(1, num_trials + 1,10):
        mean_err.append(mean[m])
        st_dev.append(stdev[m])
    if plot_SD:
        plt.errorbar(np.arange(1, num_trials + 1,10), mean_err, yerr=st_dev,   alpha=0.85,capthick=5, capsize=10, 
        linewidth=0,color=color,elinewidth=3)
    
plt.xlabel('Number of iterations')
plt.ylabel('Objective value (normalized)')
plt.title('Sampled values over time')
plt.ylim([0.2, 1.1])
plt.yticks(np.arange(0.2,1.2,0.2))
plt.tight_layout()
plt.legend(plot_leg, loc='lower right')
plt.savefig('poly_samples.pdf',dpi=1200,bbox_inches = 'tight', pad_inches = 0.1)
plt.close('all')
