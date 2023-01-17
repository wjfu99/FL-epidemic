import numpy as np
import random

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
