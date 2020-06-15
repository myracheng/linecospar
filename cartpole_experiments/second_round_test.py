import numpy as np
import os
from cartpole import *
import time
import scipy.io as io
import itertools

# cd 
test_subject = 'myra'
res = np.load('Results/%s/sim_data_0.npy'%test_subject,allow_pickle=True)
arr = np.arange(len(res))
np.random.shuffle(arr) #shuffled the actions
# 
# res2 = res.copy()
# np.random.shuffle(res2) #shuffled the actions
res3 = res.copy()

#now display them in pairs again
for ind in arr[:5]: #shuffled
    pair = res[ind]
    state1 = pair[0]
    state2 = pair[1]

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

    pref = input("0 if you prefer the first, 1 if the second, 0.5 if same")
    
    if pref == '0.5':
        res3[ind][2] = float(pref) 
    else:
        res3[ind][2] = int(pref) 

np.save('Results/%s/new_sim_data.npy'%test_subject,res3)