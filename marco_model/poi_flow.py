import numpy as np

trajs = np.load('../Agent_Epi_Sim/data/beijing/processed_data/traj_mat(filled,sample).npy')
trajs = trajs[:, :14*48]
div_len = 1
loc_num = np.unique(trajs).max() + 1
trajs = np.split(trajs, div_len, axis=1)
# loc_num =
tf_mat = np.zeros((loc_num, loc_num))
for traj in trajs:
    for t in range(traj.shape[1]-1):
        for usr in range(traj.shape[0]):
            if traj[usr, t] != traj[usr, t+1]:
                tf_mat[traj[usr, t], traj[usr, t+1]] += 1
