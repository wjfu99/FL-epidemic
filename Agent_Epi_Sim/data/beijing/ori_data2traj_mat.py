import numpy as np

user_info = np.load('./processed_data/ori_data_old.npy', allow_pickle=True).item()
traj_mat = []
for info in user_info:
    traj_mat.append(user_info[info]['trace'])

traj_mat = np.array(traj_mat)
np.save('./processed_data/traj_mat.npy', traj_mat)