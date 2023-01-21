import networkx as nx
import pickle
import csv
import torch
import numpy as np
from torch.utils.data import Dataset


class LoadData(Dataset):  # 这个就是把读入的数据处理成模型需要的训练数据和测试数据，一个一个样本能读取出来
    def __init__(self, history_length, train_mode, device):
        # 在此处读取特征文件，构造edge_index,edge_attr,flow_data
        self.train_mode = train_mode
        self.history_length = history_length


        self.edge_attr = torch.tensor(np.load('./reg_edge_att.npy')).to(device)
        self.edge_index = torch.tensor(np.load('./reg_edge_idx.npy')).to(device)
        node_feature = np.load('../Agent_Epi_Sim/data/beijing/processed_data/region_epi_freq.npy')
        node_feature = node_feature[:, :, None]
        self.dataset_len = node_feature.shape[1]
        self.flow_norm, self.flow_data = self.pre_process_data(data=node_feature, norm_dim=1)  # self.flow_norm为归一化的

    def __len__(self):
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == "train":
            return self.dataset_len - self.history_length  # 训练的样本数　＝　训练集总长度　－　历史数据长度
        elif self.train_mode == "test":
            return self.dataset_len - self.history_length + 1  # 每个样本都能测试，测试样本数　＝　测试总长度
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # 取每一个样本 (x, y), index = [0, L1 - 1]这个是根据数据集的长度确定的
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """
        # if self.train_mode == "train":
        #     index = index  # 训练集的数据是从时间０开始的，这个是每一个流量数据，要和样本（ｘ,y）区别
        # elif self.train_mode == "test":
        #     index += self.train_days * self.one_day_length  # 有一个偏移量
        # else:
        #     raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)  # 这个就是样本（ｘ,y）

        data_x = LoadData.to_tensor(data_x)  # [N, H, D] # 转换成张量
        data_y = LoadData.to_tensor(data_y).unsqueeze(1) # [N, 1, D]　# 转换成张量，在时间维度上扩维

        return {"edge_attr": self.edge_attr, "edge_index": self.edge_index, "flow_x": data_x, "flow_y": data_y}  # 组成词典返回
        # return {"flow_x": data_x, "flow_y": data_y}  # 组成词典返回

    @staticmethod
    def slice_data(data, history_length, index, train_mode):  # 根据历史长度,下标来划分数据样本
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index  # 开始下标就是时间下标本身，这个是闭区间
            end_index = index + history_length  # 结束下标,这个是开区间
        elif train_mode == "test":
            start_index = index  # 开始下标
            end_index = index + history_length # 结束下标
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]  # 在切第二维，不包括end_index
        data_y = data[:, end_index]

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):  # 预处理,归一化
        """
        :param data: np.array,原始的交通流量数据
        :param norm_dim: int,归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            norm_base: list, [max_data, min_data], 这个是归一化的基.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = LoadData.normalize_base(data, norm_dim)  # 计算 normalize base
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)  # 归一化后的数据
        nn_data = np.isnan(norm_data).any()

        return norm_base, norm_data  # 返回基是为了恢复数据做准备的

    @staticmethod
    def normalize_base(data, norm_dim):  # 计算归一化的基
        """
        :param data: np.array, 原始的交通流量数据
        :param norm_dim: int, normalization dimension.归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D], keepdims=True就保持了纬度一致
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data  # 返回最大值和最小值

    @staticmethod
    def normalize_data(max_data, min_data, data):  # 计算归一化的流量数据，用的是最大值最小值归一化法
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        if 0 in base:
            base_test = base
            base_test[(base_test == 0)] = 1
            normalized_data = (data - mid) / base_test
        else:
            normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):  # 恢复数据时使用的，为可视化比较做准备的
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data  # 原始数据

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)
