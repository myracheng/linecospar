
import numpy as np
total = 0
total_inconsis = 0
total_diff = 0
sum_diff = 0
for name in ['xcc','eith','yang']:
    a = np.load('../Results/%s/sim_data_0.npy'%name,allow_pickle=True)
    print(np.shape(a))
    for i in range(len(a)):
        # print(np.abs(a[i][-1]-a[i][-2]))
        # print(i)
        total_diff += np.abs(a[i][-1]-a[i][-2])

    first_better = np.array(a[np.where(a[:,3] > a[:,4])[0]])
    first_len = len(np.where(first_better[:,2] == 1)[0])
    second_better = np.array(a[np.where(a[:,3] < a[:,4])[0]])
    second_len = len(np.where(second_better[:,2] == 0)[0])
    # print(np.shape())

    all_diff = np.concatenate((first_better[np.where(first_better[:,2] == 1)[0]],second_better[np.where(second_better[:,2] == 0)[0]]))
    for i in range(len(all_diff)):
        # print(i)
        # print(np.abs(all_diff[i][-1]-all_diff[i][-2]))
        sum_diff += np.abs(all_diff[i][-1]-all_diff[i][-2])

    total_inconsistent = first_len + second_len
    # print(total_inconsistent)
    # print(total_inconsistent/len(a))
    total += len(a)
    total_inconsis += total_inconsistent

print("final")
# print(total_inconsis/total)
print(total_diff/total/300)
# print(total)
print(sum_diff/total_inconsis/300)
# print(total_inconsis)