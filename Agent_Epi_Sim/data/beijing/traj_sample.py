import numpy as np
# 用于在大型的数据集上直接sample出来一个小的数据集
trajs = np.load('../beijing/processed_data/traj_mat(filled,uncluster).npy')
sample_trajs = trajs[:15000, :]
np.save('../beijing/processed_data/traj_mat(filled,uncluster,sample).npy', sample_trajs)
# np.take_along_axis