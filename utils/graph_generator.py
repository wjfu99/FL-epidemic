import numpy as np
import torch
from tqdm import tqdm

def hypergraph_generator(traj, unique_len=None):
    hyperedge_index = [[], []]
    if unique_len is None:
        for uid, usr in enumerate(traj):
            loc = list(np.unique(usr))
            if -1 in loc:
                loc.remove(-1)
            hyperedge_index[0].extend([uid for i in range(len(loc))])
            hyperedge_index[1].extend(loc)
    else:
        # loc_num = loc.max()
        assert traj.shape[1] % unique_len == 0
        unique_num = traj.shape[1] // unique_len
        hyperedge2st = {}
        edge_idx = 0
        for t_idx in tqdm(range(unique_num)):
            traj_intv = traj[:, t_idx*unique_len:(t_idx+1)*unique_len]
            for usr in range(traj.shape[0]):
                usr_locs = list(set(traj_intv[usr]))
                if -1 in usr_locs:
                    usr_locs.remove(-1)
                for usr_loc in usr_locs:
                    if (usr_loc, t_idx) in hyperedge2st:
                        usr_edge_idx = hyperedge2st[(usr_loc, t_idx)]
                    else:
                        hyperedge2st[(usr_loc, t_idx)] = edge_idx
                        usr_edge_idx = edge_idx
                        edge_idx += 1
                    hyperedge_index[0].append(usr)
                    hyperedge_index[1].append(usr_edge_idx)
    hyperedge_index = np.array(hyperedge_index)
    return hyperedge_index, hyperedge2st

def hypergraph_sequence_generator(traj, seq_num, device, unique_len=48):
    trajs = np.split(traj, seq_num, axis=1)
    # hyperedge_index_list = [hypergraph_generator(i, unique_len=unique_len) for i in trajs]
    hyperedge_index_list = []
    st2idx_list = []
    for traj in trajs:
        hyperedge_index, hyperedge2st = hypergraph_generator(traj, unique_len=unique_len)
        hyperedge_index_list.append(hyperedge_index)
        st2idx_list.append(hyperedge2st)
    hyperedge_index_seq = []
    for hyperedge_index in hyperedge_index_list:
        hyperedge_index_seq.append(torch.tensor(hyperedge_index).to(device))
    return hyperedge_index_seq, st2idx_list

def hypergraph2hyperindex(hypergraph,device):
    hyperedge_index = [[], []]
    for edge_idx in range(hypergraph.shape[1]):
        node = np.argwhere(hypergraph[:, edge_idx]!=0)
        hyperedge_index[0].extend(np.squeeze(node))
        hyperedge_index[1].extend([edge_idx for _ in range(len(node))])
    hyperedge_index = np.array(hyperedge_index)
    hyperedge_index = torch.tensor(hyperedge_index).to(device)
    return hyperedge_index

