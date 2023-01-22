import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
# from modules import base_models
from . import base_models
from modules.utils.clients_dp import Dp_Agg

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        
    def forward(self, loc_emb, graph):
        return nn


    
class MultiScaleFedGNN(nn.Module):
    def __init__(self, usr_num, usr_dim=32, loc_dim_1=64, loc_dim_2=64, usr_dim1=32, usr_dim2=16, rnn_lay_num=1, class_num=2, **model_args):
        super().__init__()
        self.usr_num = usr_num
        # init user embedding
        self.usr_emb = nn.Embedding(usr_num, usr_dim)
        # self.usr_emb = nn.Parameter(torch.FloatTensor(usr_num, usr_dim))
        # self.usr_emb = nn.Linear(usr_num, usr_dim)

        # create user graph,3 the graph should be as a input to forward.
        # define the convolutional layer
        self.hyper_convs = nn.ModuleList([base_models.Dp_HypergraphConv(32, 32, **model_args), base_models.Dp_HypergraphConv(32, 32, **model_args)])
        # self.server_loc_gcn = getattr(base_models, 'GCN')(loc_dim_1, 64, loc_dim_2)
        self.loc_dp = model_args['loc_dp']
        if model_args['loc_dp']:
            self.fake_locs = model_args['fake_locs']
            self.real_locs = model_args['real_locs']
        self.clients_rnn = getattr(base_models, 'LSTM')(usr_dim1, usr_dim2, rnn_lay_num, bias=True, batch_first=False)
        # self.clients_rnn = getattr(base_models, 'Transformer')(10, 5, usr_dim1, 1, usr_dim2, n_decoder_layers=3, n_encoder_layers=3,
        #          n_heads=3)
        if model_args['macro']:
            self.macro = True
            self.macro_emb_seq = model_args['loc_emb_seq']
        else:
            self.macro = False
        self.output_layer = nn.Linear(usr_dim2, class_num)

    # TODO: emphasize that we add noise on the updated values of usr embedding.
    # TODO: notice that we can add noise to the xxx
    def forward(self, hyperedge_seq, epoch, loc_graph_idx=None):
        # usr emb add noise
        # out = self.usr_emb(range(self.usr_num))
        usr_input = self.usr_emb.weight
        outseq = None
        # x = usr_input
        # aggregate from node to edge in server
        for idx in range(len(hyperedge_seq)):
            hyperedge_idx = hyperedge_seq[idx]
            # aggregate from neighbor locations
            x = usr_input
            for hyper_conv in self.hyper_convs:
                if self.loc_dp:
                    dp_args = {'fake_loc': self.fake_locs[idx],
                               'real_loc': self.real_locs[idx],
                               'epoch': epoch
                               }
                else:
                    dp_args = None
                if self.macro:
                    macro_emb = self.macro_emb_seq[idx]
                else:
                    macro_emb = None
                x = hyper_conv(x, hyperedge_idx, dp_args=dp_args, macro_emb=macro_emb)
                x = F.relu(x)

            usr_emb = x.unsqueeze(0)
            if outseq is None:
                outseq = usr_emb
            else:
                outseq = torch.cat((outseq, usr_emb), dim=0)

        # process with RNN-based model
        out, (h, c) = self.clients_rnn(outseq)
        # outseq = outseq.permute(1, 0, 2)
        # outseq = outseq[None, :, :, :]
        # out = self.clients_rnn(outseq)
        out = F.relu(out)
        out = self.output_layer(out[-1, :, :]) #Utilize the last output of RNN for prediction.
        return out
#
# class RNN(torch.nn.Module):
#     """
#     Base class for Recurrent Neural Networks (LSTM or GRU).
#     Initialization to this class contains all variables for variation of the model.
#     Consists of one of the above RNN architectures followed by an optional GNN on the final hidden state.
#     Parameters:
#         node_features: int - number of features per node
#         output: int - length of the output vector on each node
#         dim: int - number of features of embedding for each node
#         module: torch.nn.Module - to be used in the LSTM to calculate each gate
#     """
#     def __init__(self, node_features=1, output=1, dim=32, module=GraphLinear, rnn=LSTM, gnn=WeightedSAGEConv, gnn_2=WeightedSAGEConv, rnn_depth=1, name="RNN", edge_count=423, skip_connection=True):
#         super(RNN, self).__init__()
#         self.dim = dim
#         self.rnn_depth = rnn_depth
#         self.name = name
#         self.skip_connection = skip_connection
#
#         # Ensure that matrix multiplication sizes match up based on whether GNNs and RNN are used
#         if gnn:
#             if skip_connection:
#                 self.gnn = gnn(node_features, dim)
#             else:
#                 self.gnn = gnn(node_features, dim * 2)
#             if rnn:
#                 if skip_connection:
#                     self.recurrent = rnn(dim, dim, module=module)
#                 else:
#                     self.recurrent = rnn(dim * 2, dim * 2, module=module)
#             else:
#                 self.recurrent = None
#         else:
#             self.gnn = None
#             if rnn:
#                 self.recurrent = rnn(node_features, dim, module=module)
#             else:
#                 self.recurrent = None
#         if gnn_2:
#             if gnn:
#                 self.gnn_2 = gnn_2(dim * 2, dim * 2)
#             else:
#                 self.gnn_2 = gnn_2(dim + node_features, dim * 2)
#         else:
#             self.gnn_2 = None
#
#         self.lin1 = torch.nn.Linear(2 * dim, dim)
#         self.lin2 = torch.nn.Linear(dim, output)
#         self.act1 = torch.nn.ReLU()
#         self.act2 = torch.nn.ReLU()
#
#     def forward(self, data, h=None, c=None):
#         # Get data from snapshot
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#
#         # First GNN Layer
#         if self.gnn:
#             x = self.gnn(x, edge_index, edge_attr)
#             x = F.relu(x)
#
#         # Initialize hidden and cell states if None
#         current_dim = self.dim
#         if not self.skip_connection:
#             current_dim = self.dim * 2
#         if h is None:
#             h = torch.zeros(x.shape[0], current_dim)
#         if c is None:
#             c = torch.zeros(x.shape[0], current_dim)
#
#         # RNN Layer
#         if self.recurrent:
#             for i in range(self.rnn_depth):
#                 h, c = self.recurrent(x, edge_index, edge_attr, h, c)
#
#         # Skip connection from first GNN
#         if self.skip_connection:
#             x = torch.cat((x, h), 1)
#         else:
#             x = h
#
#         # Second GNN Layer
#         if self.gnn_2:
#             x = self.gnn_2(x, edge_index, edge_attr)
#
#         # Readout and activation layers
#         x = self.lin1(x)
#         # x = self.act1(x)
#         x = self.lin2(x)
#         # x = self.act2(x)
#
#         return x, h, c
