import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, TransformerConv, SAGEConv, SuperGATConv, ClusterGCNConv, GCN2Conv, AGNNConv, TAGConv
#
class GCN(nn.Module):
    def __init__(self, usr_num, usr_dim=32):
        super().__init__()
        self.usr_num = usr_num
        # init user embedding
        self.usr_emb = nn.Embedding(usr_num, usr_dim)
        conv = TransformerConv
        self.conv1 = conv(32, 32, heads=1)
        self.conv2 = conv(32, 16, heads=1)
        self.output = nn.Linear(16, 2)
    def forward(self, edge_index):
        usr_input = self.usr_emb.weight

        out = self.conv1(usr_input, edge_index)
        out = F.relu(out)
        out = self.conv2(out, edge_index)
        out = F.relu(out)
        out = self.output(out)
        return out

# class GCN(nn.Module):
#     def __init__(self, usr_num, usr_dim=32):
#         super().__init__()
#         self.usr_num = usr_num
#         # init user embedding
#         self.usr_emb = nn.Embedding(usr_num, usr_dim)
#
#         self.conv1 = GCNConv(32, 32)
#         self.conv2 = GCNConv(32, 16)
#         self.output = nn.Linear(16, 2)
#     def forward(self, edge_index):
#         usr_input = self.usr_emb.weight
#
#         out = self.conv1(usr_input, edge_index)
#         out = F.relu(out)
#         out = self.conv2(out, edge_index)
#         out = F.relu(out)
#         out = self.output(out)
#         return out