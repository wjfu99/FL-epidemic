import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

# Following codes is forked form torch_geometric.
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hypergraph_conv.html#HypergraphConv


class Hcov_node2edge(MessagePassing):
    def __init__(self):
        super(Hcov_node2edge, self).__init__()
    # TODO: 1. try to introduce the attention
    # TODO: 2 hyperedge_weight suit for adjusting weights between nodes and edges
    def forward(self, x, hyperedge_index , hyperedge_weight=None):
        
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        # x = self.lin(x)
        alpha = None
        # D = scatter_add(hyperedge_weight[hyperedge_index[1]],
        #                 hyperedge_index[0], dim=0, dim_size=num_nodes)
        # D = 1.0 / D
        # D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        # out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
        #                      alpha=alpha, size=(num_edges, num_nodes))
        return out


class Hcov_edge2node(MessagePassing):
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        super(Hcov_edge2node, self).__init__()
        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')
    def forward(self, x, hyperedge_index , hyperedge_weight=None):
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        # x = self.lin(x)
        alpha = None
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0
        #
        # B = scatter_add(x.new_ones(hyperedge_index.size(1)),
        #                 hyperedge_index[1], dim=0, dim_size=num_edges)
        # B = 1.0 / B
        # B[B == float("inf")] = 0

        # out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
        #                      size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=x, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))
        out = self.lin(out)
        return out