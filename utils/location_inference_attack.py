import numpy as np
import os
from tqdm import tqdm



def global_tf_mat(traj_mat):
    trajs = traj_mat
    # div_len = 1
    loc_num = np.unique(trajs).max() + 1
    # trajs = np.split(trajs, div_len, axis=1)
    # loc_num =
    tf_mat = np.zeros((loc_num, loc_num))
    for t in tqdm(range(trajs.shape[1] - 1)):
        for usr in range(trajs.shape[0]):
            # if trajs[usr, t] != trajs[usr, t + 1]:
            tf_mat[trajs[usr, t], trajs[usr, t + 1]] += 1
    tf_sum = tf_mat.sum(axis=1)
    tf_mat = tf_mat / tf_sum[:, np.newaxis]
    tf_mat = np.nan_to_num(tf_mat, nan=1 / loc_num)
    return tf_mat


generator = "uni_iid"
data_path = '../datasets/beijing/large-filled-clustered/'
files = os.listdir(data_path + generator)
traj_mat = np.load(data_path + "traj_mat(filled,sample).npy")
traj_mat = traj_mat[:, :14*48]
fake_mats = []
for file in files:
    fake = np.load(data_path + generator + '/' + file)
    fake_mats.append(fake)

# tf_mat = global_tf_mat(traj_mat)
# np.save(data_path + "tf_mat.npy", tf_mat)

tf_mat = np.load(data_path + "tf_mat.npy")
fakes_num = 5
usr_num = 10000

wrong_cnt = 0
for uid in range(usr_num):
    real = traj_mat[uid][0:3]
    fakes = [fake_mat[uid][0:3] for fake_mat in fake_mats]
    fakes.insert(0, real)
    all = fakes
    prob_max = 0
    flag = False
    for i in range(fakes_num+1):
        for j in range(fakes_num+1):
            prob = tf_mat[all[i][0], all[j][1]] * tf_mat[all[i][1], all[j][2]]
            if prob > prob_max:
                prob_max = prob
                if i != 0 or j != 0:
                    flag = True
                    wrong_cnt += 1
                    break
        if flag:
            break

privacy = wrong_cnt / usr_num
