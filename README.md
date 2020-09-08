## LineCoSpar
This repository contains the source code for the LineCoSpar algorithm and implementation of the experiments described in the paper:

**Human Preference-Based Learning for High-dimensional Optimization of Exoskeleton Walking Gaits.** 

[Paper](https://arxiv.org/pdf/2003.06495.pdf) &nbsp; &nbsp; &nbsp;  [Video](https://youtu.be/c6a0kXMyML0)

LineCoSpar is a human-in-the-loop preference-based algorithm that enables optimization over many parameters by iteratively exploring one-dimensional subspaces. LineCoSpar is an extension of [CoSpar](https://github.com/ernovoseller/CoSpar). The high-dimensional performance of LineCoSpar is demonstrated through both simulations and human-subject trials. The human-subject trials in the IROS publication consisted of optimizing 6 walking gait parameters of a lower-body exoskeleton for 6 able-bodied subjects. Our analysis of the lower-body exoskeleton experiments highlights differences in the utility functions underlying individual users' gait preferences. 

___

## Respository Contents
- `exoskeleton_pref_learning.py`: version of the LineCoSpar algorithm that was deployed on the lower-body exoskeleton for the IROS publication experiments.
- `Plotting`: code to plot the simulation results
- `cartpole_experiments`: code to run simulated experiments of LineCoSpar on a cart-pole system. This set of simulated experiments, which was not discussed in the IROS publication, provide another example of how the algorithm can be used.
- `gaitAnalysis` folder: code to fit various cost function terms to the lower-body exoskeleton experimental preferences. This folder also contains code to plot the CoM and CoP trajectories of the most- and least-preferred gaits.
- `synthetic_fns`: code for simulations on the synthetic functions used in the IROS publication.

