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
            hyperedge_index[0].extend([uid for i in range(unique_num(loc))])
            hyperedge_index[1].extend(loc)
    else:
        locs = list(np.unique(traj))
        if -1 in locs:
            locs.remove(-1)
        # loc_num = loc.max()
        assert traj.shape[1] % unique_len ==0
        unique_num = traj.shape[1] // unique_len
        for t_idx in tqdm(range(unique_num)):
            traj_intv = traj[:, t_idx*unique_len:(t_idx+1)*unique_len]
            for loc in locs:
                same = np.argwhere(traj_intv == loc)
                same = same[:, 0]
                same = np.unique(same)
                hyperedge_index[0].extend(list(same))
                hyperedge_index[1].extend([loc for i in range(len(list(same)))])



    hyperedge_index = np.array(hyperedge_index)
    return hyperedge_index

def hypergraph_sequence_generator(traj, seq_num, device):
    trajs = np.split(traj, seq_num, axis=1)
    hyperedge_index_seq = [torch.tensor(hypergraph_generator(i, unique_len=2)).to(device) for i in trajs]
    return hyperedge_index_seq

def hypergraph2hyperindex(hypergraph,device):
    hyperedge_index = [[], []]
    for edge_idx in range(hypergraph.shape[1]):
        node = np.argwhere(hypergraph[:, edge_idx]!=0)
        hyperedge_index[0].extend(np.squeeze(node))
        hyperedge_index[1].extend([edge_idx for _ in range(len(node))])
    hyperedge_index = np.array(hyperedge_index)
    hyperedge_index = torch.tensor(hyperedge_index).to(device)
    return hyperedge_index

