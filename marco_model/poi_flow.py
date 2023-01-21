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

edge_index = [[], []]
edge_attr = []
for i in range(tf_mat.shape[0]):
    for j in range(tf_mat.shape[1]):
        if tf_mat[i, j] != 0:
            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_attr.append(tf_mat[i, j])
edge_index = np.array(edge_index)
edge_attr = np.array(edge_attr)
np.save('reg_edge_idx.npy', edge_index)
np.save('reg_edge_att.npy', edge_attr)
