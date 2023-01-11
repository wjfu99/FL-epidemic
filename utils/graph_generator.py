import numpy as np
import torch

def hypergraph_generator(traj):
    hyperedge_index = [[], []]
    for uid, usr in enumerate(traj):
        loc = list(np.unique(usr))
        if -1 in loc:
            loc.remove(-1)
        hyperedge_index[0].extend([uid for i in range(len(loc))])
        hyperedge_index[1].extend(loc)
    hyperedge_index = np.array(hyperedge_index)
    return hyperedge_index

def hypergraph_sequence_generator(traj, seq_num, device):
    trajs = np.split(traj, seq_num, axis=1)
    hyperedge_index_seq = [torch.tensor(hypergraph_generator(i)).to(device) for i in trajs]
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

