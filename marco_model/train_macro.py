import torch

from torch_geometric.data import Data
from dataset import LoadData
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
from RNNModel.recurrent import RNNModel

datasets = 'basic'
dynamic = 'Omicron'

if datasets == 'basic':
    node_num = 11459
    batch_size = 37

    data_path = '../datasets/beijing/Basic/'
    if dynamic == 'SARS-CoV-2':
        node_feature = np.load(data_path + 'region_epi_freq.npy')
    elif dynamic == 'Omicron':
        node_feature = np.load(data_path + 'region_epi_freq_omicron.npy')
elif datasets == 'larger':
    node_num = 3578
    batch_size = 11

    data_path = '../datasets/beijing/Larger/'
    if dynamic == 'SARS-CoV-2':
        node_feature = np.load(data_path + 'region_epi_freq.npy')
    elif dynamic == 'Omicron':
        node_feature = np.load(data_path + 'region_epi_freq_omicron.npy')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    edge_index = torch.tensor(np.load('./reg_edge_idx.npy'))
    edge_weight = torch.tensor(np.load('./reg_edge_att.npy'))
    my_net = RNNModel(sparse_idx=edge_index, edge_weights=edge_weight, conv_method='GConvGRU', max_view=1,
                      node_num=node_num, layer_num=1, input_dim=1, output_dim=1, seq_len=3, horizon=1).to(device)

    train_data = LoadData(history_length=3, train_mode="train", node_feature=node_feature)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_data = LoadData(history_length=3, train_mode="test", node_feature=node_feature)
    test_loader = DataLoader(test_data, batch_size=batch_size + 1, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    Epoch = 100
    my_net.train()
    Train_loss = []
    for epoch in range(Epoch):
        start_time = time.time()
        epoch_loss = 0.0
        for idex, data in enumerate(train_loader):
            input = data["flow_x"].to(device)
            target = data["flow_y"].to(device)
            input = input.permute(2, 0, 1, 3)
            target = target.permute(2, 0, 1, 3)
            predict_value, encoder_hidden_state = my_net(input)
            loss = criterion(predict_value, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
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
                                                            test_loader)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

            print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))
        torch.save(my_net.state_dict(), 'mainmodel.pt')
    print("Performance:  MAE {:2.2f}    {:2.2f}%    {:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))


def compute_performance(prediction, target, data):
    try:
        dataset = data.dataset
    except:
        dataset = data


    print("prediction：", prediction.size())

    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1],
                                       prediction.data.cpu().numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1],
                                   target.data.cpu().numpy())  # [5886,16,1]

    mae, mape, rmse = Evaluation.total(target, prediction)

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data


class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 5))

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
