import numpy as np
from tqdm import tqdm


trajs = np.load('../Agent_Epi_Sim/data/beijing/processed_data/traj_mat.npy')
trajs = trajs[:, :40*48]

div_len = 1
loc_num = np.unique(trajs).max() + 1

tf_mat = np.zeros((loc_num, loc_num))
for uid in tqdm(range(trajs.shape[0])):
    last = None
    for t in range(trajs.shape[1]):
        if trajs[uid][t] != -1:
            if last is not None:
                tf_mat[last, trajs[uid][t]] += 1
            last = trajs[uid][t]




edge_index = [[], []]
edge_attr = []
for i in tqdm(range(tf_mat.shape[0])):
    for j in range(tf_mat.shape[1]):
        if tf_mat[i, j] != 0:
            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_attr.append(tf_mat[i, j])
edge_index = np.array(edge_index)
edge_attr = np.array(edge_attr)
np.save('reg_edge_idx.npy', edge_index)
np.save('reg_edge_att.npy', edge_attr)
