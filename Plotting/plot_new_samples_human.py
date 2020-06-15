
# """
# Plot sampled values in cartpole environment.
# """
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
    "figure.figsize": [18,9],     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",    # use utf8 input and T1 fonts 
        r"\usepackage[T1]{fontenc}",        # plots will be generated 
        r"\usepackage{amsmath}",
        r"\usepackage{bm}"
        ]                                   # using this preamble
    }
mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)
plt.figure(figsize = (18, 11))
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Color-blind friendly palette: https://gist.github.com/thriveth/8560036
CB_colors = ['#377eb8', '#4daf4a', '#ff7f00', 
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']                  

colors = CB_colors

num_runs = 1     # Times experiment was repeated
num_trials = 100 # Total number of posterior samples/trials


alpha = 0.4

dirs = ["../Results/xcc","../eith","../yang"]

plot_legs = ["Human feedback","Simulated feedback"]
color_count = 0
plt.close('all')
all_obj_vals = []
for d in dirs:
    os.chdir(d)
    obj_vals = np.empty((num_trials,0))

    for filename in glob.glob("*.mat"):
        results = io.loadmat(filename)
        obj_vals =  np.array(results['objective_values'])
        all_obj_vals.append(np.mean(obj_vals,axis=1))

    plt.xlabel('Number of iterations')
    plt.ylabel('Objective function value')

all_obj_vals = np.array(all_obj_vals)/300
num_to_plot = num_trials
mean_linewidth = 2
mean = np.mean(all_obj_vals,axis=0)
stdev = np.std(all_obj_vals,axis=0)
plt.errorbar(np.arange(1, num_trials + 1), mean,linewidth=5,color=colors[0],label=plot_legs[0])
mean_err = []
st_dev = []
for m in np.arange(0, num_trials,5):
    mean_err.append(mean[m])
    st_dev.append(stdev[m])
plt.errorbar(np.arange(1, num_trials+1,5), mean_err, yerr=st_dev,   alpha=0.85, capthick=5,capsize=10, 
    linewidth=0,color=colors[0],
        elinewidth=3)

dirs = ["../testpd_comp_531"]
all_obj_vals = []
for d in dirs:
    os.chdir(d)
    obj_vals = np.empty((num_trials,0))

    for filename in glob.glob("*.mat"):
        results = io.loadmat(filename)
        obj_vals =  np.array(results['objective_values'])[:num_trials]
        all_obj_vals.append(np.mean(obj_vals,axis=1))

all_obj_vals = np.array(all_obj_vals)/300
num_to_plot = num_trials
mean_linewidth = 2
mean = np.mean(all_obj_vals,axis=0)
stdev = np.std(all_obj_vals,axis=0)
plt.errorbar(np.arange(1, num_trials + 1), mean,linewidth=5,color=colors[1],label=plot_legs[1])
mean_err = []
st_dev = []
for m in np.arange(0, num_trials,5):
    mean_err.append(mean[m])
    st_dev.append(stdev[m])
plt.errorbar(np.arange(1, num_trials+1,5), mean_err, yerr=st_dev,   alpha=0.85, capthick=5,capsize=10, color=colors[1],
    linewidth=0,
        elinewidth=3)

plt.xlabel(r'Number of iterations ($t$)')
plt.ylabel(r'Reward ($r$, normalized)')
plt.title("Sampled values over time")
plt.ylim([-0.1, 1.2])
plt.yticks(np.arange(0,1.1,0.2))
plt.tight_layout() 
plt.legend()
plt.savefig('average_all_samples_100.pdf',dpi=1200,bbox_inches = 'tight', pad_inches = 0.1)
plt.close('all')
