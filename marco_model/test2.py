import torch
from torch.nn import Parameter
from weight_sage import WeightedSAGEConv
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from graph_nets import GraphLinear
from torch_geometric.data import Data
from dataset import LoadData
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import graph_nets
import time
import numpy as np
from RNNModel.recurrent import RNNModel

# CONSTRUCT MODELS
WSC = WeightedSAGEConv
USC = lambda in_channels, out_channels, bias=True: WeightedSAGEConv(in_channels, out_channels, weighted=False)
linear_module = lambda in_channels, out_channels, bias: graph_nets.GraphLinear(in_channels, out_channels, bias=bias)
DeepUSC = lambda lookback, dim: graph_nets.GNNModule(USC, 3, lookback, dim=dim, res_factors=[1, 0, 1], dropouts=[1])
DeepWSC = lambda lookback, dim: graph_nets.GNNModule(WSC, 3, lookback, dim=dim, res_factors=[1, 0, 1], dropouts=[1])


# Recurrent Neural Network Modules
class LSTM(torch.nn.Module):
    # This is an adaptation of torch_geometric_temporal.nn.GConvLSTM, with ChebConv replaced by the given model.
    """
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        module (torch.nn.Module, optional): The layer or set of layers used to calculate each gate.
            Could also be a lambda function returning a torch.nn.Module when given the parameters in_channels: int, out_channels: int, and bias: bool
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(LSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = WeightedSAGEConv(self.in_channels, self.out_channels, self.bias)

        self.conv_h_i = WeightedSAGEConv(self.out_channels, self.out_channels, self.bias)

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = WeightedSAGEConv(self.in_channels, self.out_channels, self.bias)

        self.conv_h_f = WeightedSAGEConv(self.out_channels, self.out_channels, self.bias)

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = WeightedSAGEConv(self.in_channels, self.out_channels, self.bias)

        self.conv_h_c = WeightedSAGEConv(self.out_channels, self.out_channels, self.bias)

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = WeightedSAGEConv(self.in_channels, self.out_channels, self.bias)

        self.conv_h_o = WeightedSAGEConv(self.out_channels, self.out_channels, self.bias)

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
        I = self.conv_x_i(X, edge_index, edge_weight)
        I = I + self.conv_h_i(H, edge_index, edge_weight)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = self.conv_x_f(X, edge_index, edge_weight)
        F = F + self.conv_h_f(H, edge_index, edge_weight)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_weight)
        T = T + self.conv_h_c(H, edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C):
        O = self.conv_x_o(X, edge_index, edge_weight)
        O = O + self.conv_h_o(H, edge_index, edge_weight)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None, C: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C


class RNN(torch.nn.Module):
    """
    Base class for Recurrent Neural Networks (LSTM or GRU).
    Initialization to this class contains all variables for variation of the model.
    Consists of one of the above RNN architectures followed by an optional GNN on the final hidden state.
    Parameters:
        node_features: int - number of features per node
        output: int - length of the output vector on each node
        dim: int - number of features of embedding for each node
        module: torch.nn.Module - to be used in the LSTM to calculate each gate
    """

    def __init__(self, node_features=1, output=1, dim=32, rnn_depth=1):
        super(RNN, self).__init__()
        self.node_features = node_features
        self.dim = dim
        self.rnn_depth = rnn_depth

        # Ensure that matrix multiplication sizes match up based on whether GNNs and RNN are used
        self.gnn = DeepWSC(self.node_features, dim)
        self.recurrent = LSTM(self.dim, self.dim)
        self.gnn_2 = DeepWSC(dim * 2, dim * 2)

        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, output)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data, h=None, c=None):
        # Get data from snapshot
        x, edge_index, edge_attr = data.x, data.edge_index[0], data.edge_attr[0]

        # First GNN Layer
        x = self.gnn(x, edge_index, edge_attr)
        x = F.relu(x)

        # Initialize hidden and cell states if None
        current_dim = self.dim

        if h is None:
            h = torch.zeros(x.shape[1], x.shape[2], current_dim)
        if c is None:
            c = torch.zeros(x.shape[1], x.shape[2], current_dim)

        # RNN Layer
        for i in range(self.rnn_depth):
            h, c = self.recurrent(x, edge_index, edge_attr, h, c)

        # Skip connection from first GNN
        x = torch.cat((x, h), 3)

        # Second GNN Layer
        x = self.gnn_2(x, edge_index, edge_attr)

        # Readout and activation layers
        x = self.lin1(x)
        # x = self.act1(x)
        x = self.lin2(x)
        # x = self.act2(x)

        return x, h, c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    edge_index = torch.tensor(np.load('./reg_edge_idx.npy'))
    edge_weight = torch.tensor(np.load('./reg_edge_att.npy'))
    my_net = RNNModel(sparse_idx=edge_index, edge_weights=edge_weight, conv_method='GConvGRU', max_view=1,
                      node_num=3578, layer_num=1, input_dim=1, output_dim=1, seq_len=3, horizon=1).to(device)

    train_data = LoadData(history_length=3, train_mode="train", device=device)
    train_loader = DataLoader(train_data, batch_size=11, shuffle=False)
    test_data = LoadData(history_length=3, train_mode="test", device=device)
    test_loader = DataLoader(test_data, batch_size=12, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.005)  # 没写学习率，表示使用的是默认的，也就是lr=1e-3
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    Epoch = 10  # 训练的次数
    my_net.train()  # 打开训练模式
    Train_loss = []
    Train_loss = []
    for epoch in range(Epoch):
        start_time = time.time()
        epoch_loss = 0.0
        for idex, data in enumerate(train_loader):
            input = data["flow_x"].to(device)
            edge_index = data["edge_index"].to(device)
            edge_attr = data["edge_attr"].to(device)
            target = data["flow_y"].to(device)
            data = Data(input, edge_index, edge_attr)
            input = input.permute(2, 0, 1, 3)
            target = target.permute(2, 0, 1, 3)
            predict_value, encoder_hidden_state = my_net(input)
            loss = criterion(predict_value, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()  # 更新参数
            end_time = time.time()
            b = 1000 * epoch_loss / len(train_data)
            Train_loss.append(b)
        print(
            "Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                            (end_time - start_time) / 60))

    # Test Model
    my_net.eval()
    with torch.no_grad():
        MAE, MAPE, RMSE = [], [], []

        total_loss = 0.0
        Test_loss = []
        for idex, data in enumerate(test_loader):
            # 下面得到的预测结果是归一化的结果
            input = data["flow_x"].to(device)
            edge_index = data["edge_index"].to(device)
            edge_attr = data["edge_attr"].to(device)
            target = data["flow_y"].to(device)
            data = Data(input, edge_index, edge_attr)
            input = input.permute(2, 0, 1, 3)
            target = target.permute(2, 0, 1, 3)
            predict_value, encoder_hidden_state = my_net(input)
            loss = criterion(predict_value, target)
            Test_loss.append(loss.item())
            total_loss += loss.item()

            performance, data_to_save = compute_performance(predict_value, target,
                                                            test_loader)  # 计算模型的性能，返回评价结果和恢复好的数据

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

            print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))
        torch.save(my_net.state_dict(), 'mainmodel.pt')

    # 三种指标取平均
    print("Performance:  MAE {:2.2f}    {:2.2f}%    {:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))


def compute_performance(prediction, target, data):  # 计算模型性能
    # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
    # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
    try:
        dataset = data.dataset
    except:
        dataset = data

    # 对预测和目标数据进行逆归一化,flow_norm为归一化的基，flow_norm[0]为最大值，flow_norm[1]为最小值
    print("prediction：", prediction.size())

    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1],
                                       prediction.data.cpu().numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1],
                                   target.data.cpu().numpy())  # [5886,16,1]

    mae, mape, rmse = Evaluation.total(target, prediction)  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）


class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 5))  # 加５是因为target有可能为0，当然只要不太大，加几都行

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)

        return mae, mape, rmse


if __name__ == '__main__':
    main()
