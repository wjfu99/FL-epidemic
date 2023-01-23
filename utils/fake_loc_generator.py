import os.path

import numpy as np
import random
from tqdm import tqdm

# A simple random fake location generator
def fake_loc_gen(traj_mat, seq_num):
    trajs = np.split(traj_mat, seq_num, axis=1)
    real_locs = []
    fake_locs = []
    loc_set = set(np.unique(traj_mat))
    for i, traj in enumerate(trajs):
        real_locs.append({})
        fake_locs.append({})
        for uid, usr in enumerate(traj):
            r_loc = set(np.unique(usr))
            if -1 in r_loc:
                r_loc.remove(-1)
            f_loc = random.choices(list(loc_set - r_loc), k=len(r_loc))
            real_locs[i][uid] = r_loc
            fake_locs[i][uid] = f_loc
    return real_locs, fake_locs

def global_tf_mat(traj_mat):
    trajs = traj_mat
    # div_len = 1
    loc_num = np.unique(trajs).max() + 1
    # trajs = np.split(trajs, div_len, axis=1)
    # loc_num =
    tf_mat = np.zeros((loc_num, loc_num))
    for t in range(trajs.shape[1] - 1):
        for usr in range(trajs.shape[0]):
            # if trajs[usr, t] != trajs[usr, t + 1]:
            tf_mat[trajs[usr, t], trajs[usr, t + 1]] += 1
    tf_sum = tf_mat.sum(axis=1)
    tf_mat = tf_mat / tf_sum[:, np.newaxis]
    tf_mat = np.nan_to_num(tf_mat, nan=1 / loc_num)
    return tf_mat

def eval_epi_domain(epi_risk):
    return epi_risk

def plausible_loc_gen(traj_mat, seq_num, unique_len, fake_trajs_dir, epi_risk=None, index_seq=None):
    tf_mat = global_tf_mat(traj_mat)
    fake_traj_mat = np.full(traj_mat.shape, -1, dtype=int)
    epi_domain = {}
    loc_set = set(np.unique(traj_mat))
    if not os.path.isfile(fake_trajs_dir):
        for time in tqdm(range(traj_mat.shape[1])):
            epi_domain = epi_domain  # TODO xxxx
            for uid in range(traj_mat.shape[0]):
                loc = traj_mat[uid, time]
                # loc_epi_domain = epi_domain[time][loc]
                loc_epi_domain = list(loc_set)
                if time == 0:
                    fake_loc = random.choice(loc_epi_domain)
                else:
                    last_loc = fake_traj_mat[uid, time-1]
                    tf_vec = tf_mat[last_loc, loc_epi_domain]
                    tf_vec = tf_vec / tf_vec.sum()
                    fake_loc = np.random.choice(loc_epi_domain, 1, p=tf_vec)
                fake_traj_mat[uid, time] = fake_loc
        np.save(fake_trajs_dir, fake_traj_mat)
    else:
        fake_traj_mat = np.load(fake_trajs_dir)

    unique_num = traj_mat.shape[1] // seq_num // unique_len

    fake_trajs = np.split(fake_traj_mat, seq_num, axis=1)
    real_trajs = np.split(traj_mat, seq_num, axis=1)
    fake_hyperedge_index = []  # seq_num * user_num * fake_edge_num
    real_hyperedge_index = []
    for seq_idx in tqdm(range(seq_num)):
        fake_edge_index = {}
        real_edge_index = {}
        for uid in range(traj_mat.shape[0]):
            fake_edge_index[uid] = []
            real_edge_index[uid] = []
            for t_idx in range(unique_num):
                fake_traj = fake_trajs[seq_idx][uid, t_idx*unique_len:(t_idx+1)*unique_len]
                fake_locs = np.unique(fake_traj)
                real_traj = real_trajs[seq_idx][uid, t_idx * unique_len:(t_idx + 1) * unique_len]
                real_locs = np.unique(real_traj)
                for fake_loc in fake_locs:
                    # NOTE: The fake locations may not exist in the spatia-temporal hyperedge.
                    if (fake_loc, t_idx) in index_seq[seq_idx]:
                        fake_edge_index[uid].append(index_seq[seq_idx][(fake_loc, t_idx)])
                for real_loc in real_locs:
                    real_edge_index[uid].append(index_seq[seq_idx][(real_loc, t_idx)])
        fake_hyperedge_index.append(fake_edge_index)
        real_hyperedge_index.append(real_edge_index)
    return fake_hyperedge_index, real_hyperedge_index
        # for uid, usr_traj in enumerate(traj):
        #     if idx == 0:
        #         fake_locs[uid][idx] =

