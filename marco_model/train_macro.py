import torch

from torch_geometric.data import Data
from dataset import LoadData
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import graph_nets
import time
import numpy as np
from RNNModel.recurrent import RNNModel


datasets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    edge_index = torch.tensor(np.load('./reg_edge_idx.npy'))
    edge_weight = torch.tensor(np.load('./reg_edge_att.npy'))
    my_net = RNNModel(sparse_idx=edge_index, edge_weights=edge_weight, conv_method='GConvGRU', max_view=1,
                      node_num=11459, layer_num=1, input_dim=1, output_dim=1, seq_len=3, horizon=1).to(device)

    train_data = LoadData(history_length=3, train_mode="train", device=device)
    train_loader = DataLoader(train_data, batch_size=37, shuffle=False)
    test_data = LoadData(history_length=3, train_mode="test", device=device)
    test_loader = DataLoader(test_data, batch_size=38, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    Epoch = 100
    my_net.train()
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
        if 1000 * epoch_loss / len(train_data) < 6000:
            break
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
            pv = predict_value.cpu().detach().numpy()
            tv = target.cpu().detach().numpy()
            np.save('region_epi_emb.npy', encoder_hidden_state.cpu().detach().numpy())
            # loss = criterion(predict_value, target)
            # Test_loss.append(loss.item())
            # total_loss += loss.item()

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
