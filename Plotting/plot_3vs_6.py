# -*- coding: utf-8 -*-
"""
Plot results from simulations optimizing Hartmann objective functions.
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

plt.close('all')
fig = plt.figure(figsize = (18, 9))
rc('text', usetex=True)

# Color-blind friendly palette: https://gist.github.com/thriveth/8560036
CB_colors = ['#377eb8', '#4daf4a', '#ff7f00', 
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']                  

colors = CB_colors
   
fig_num = 3

num_runs = 1     # Times experiment was repeated
num_trials = 150    # Total number of posterior samples/trials


alpha = 0.4
num_samples = 3

dirs = ["../3baseline_hartmann_626", "../3randdir_hartmann_0.00random_0.0000c_302",
"../randdir_hartmann_0.00random_0.0000c_759"]
plot_legs = [
    r'\textsc{CoSpar}',
             r'\textsc{LineCoSpar} (\textsc{H3})',
             r'\textsc{LineCoSpar (\textsc{H6})}']

maxes = [3.8,3.8,3.1]
dims = [3, 3, 6]

for i, mydir in enumerate(dirs):
    os.chdir(mydir)
    obj_vals = np.empty((num_trials,0))
    for filename in glob.glob("*.mat"):
        results = io.loadmat(filename)
        obj = np.array(results['objective_values'])[:num_trials, :]
        obj =  obj/maxes[i]
        obj_vals = np.concatenate((obj_vals, obj), axis=1)
    
    mean =  np.array([np.mean(vals) for vals in obj_vals])[:num_trials]
    stdev = np.array([np.std(vals) for vals in obj_vals])[:num_trials]
    mins = np.array([np.min(vals) for vals in obj_vals])[:num_trials]
    maxs = np.array([np.max(vals) for vals in obj_vals])[:num_trials]
    
    color = colors[i]
    # mean_linestyle = 'dotted'
    mean_linewidth = 2
    plot_SD = True
    # # Plot the mean over the trials: 
    plt.errorbar(np.arange(1, num_trials + 1), mean,linewidth=5,color=color,label=plot_legs[i])

    # # Add deviation to plot
    mean_err = []
    st_dev = []
    for m in np.arange(1, num_trials + 1,10):
        mean_err.append(mean[m])
        st_dev.append(stdev[m])
    if plot_SD:
        plt.errorbar(np.arange(1, num_trials + 1,10), mean_err, yerr=st_dev,   alpha=0.85, capthick=5,capsize=10, 
linewidth=0,color=color,elinewidth=3)

plt.xlabel('Number of iterations ($t$)')
plt.ylabel('Objective value (normalized)')
plt.title('Sampled values over time')
plt.ylim([-0.1, 1.2])
plt.yticks(np.arange(0,1.1,0.2))
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig('fig1.pdf',dpi=1200,bbox_inches = 'tight', pad_inches = 0.1)
plt.close('all')
