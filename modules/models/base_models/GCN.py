import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, inductive=False, add_self_loops=True, node_dim=0, *args, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(input_size, hidden_size, cached=not inductive, add_self_loops=add_self_loops, node_dim=node_dim)
        self.conv2 = GCNConv(hidden_size, output_size, cached=not inductive, add_self_loops=add_self_loops, node_dim=node_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--hidden_size', type=int, default=128)
    #     parser.add_argument('--dropout', type=float, default=0.5)
    #     return parser