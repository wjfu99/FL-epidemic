import torch
import torch.nn as nn
import math
import numpy as np
import itertools
from collections import Counter

class Dp_Agg(nn.Module):
    def __init__(self, eps, delt, clip):
        super().__init__()
        self.eps = eps
        self.delt = delt
        self.clip = clip

    def forward(self, loc_emb, fake_loc, real_loc):
        # sigmod = self.C * math.sqrt(2 * math.log(1.25/self.delt), math.e) / self.eps
        # fake_loc = self.fake_loc
        # true_loc = self.true_loc
        # loc num counter
        # loc_cnt = [list(real) + list(fake) for real, fake in zip(real_loc.values(), fake_loc.values())]
        # loc_cnt = list(itertools.chain.from_iterable(loc_cnt))
        # loc_cnt = dict(Counter(loc_cnt))
        edge_cnt = []
        for real in real_loc.values():
            edge_cnt.extend(list(real))
        for fake in fake_loc.values():
            edge_cnt.extend(list(fake))
        edge_cnt = dict(Counter(edge_cnt))
        sigmod = self.clip * math.sqrt(2 * math.log(1.25 / self.delt, math.e)) / self.eps
        loc_noise = np.zeros(loc_emb.shape)
        for usr in real_loc:
            loc_num = len(real_loc[usr]) + len(fake_loc[usr])
            noise = np.random.normal(loc=0, scale=sigmod, size=(loc_num, loc_emb.shape[1]))
            i = 0
            for loc in real_loc[usr]:
                loc_noise[loc] += noise[i]/edge_cnt[loc]
                i += 1
            for loc in fake_loc[usr]:
                loc_noise[loc] += noise[i]/edge_cnt[loc]
                i += 1
        loc_emb_noise = loc_emb + torch.Tensor(loc_noise).to(loc_emb.device)
        return loc_emb_noise
