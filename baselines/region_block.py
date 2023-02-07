import numpy as np
from tqdm import tqdm


data_path = '../datasets/beijing/large-filled-clustered/'
trace_array = np.load(data_path + "traj_mat(filled,sample).npy")
trace_array = trace_array[:, 14:48]
pop_num = trace_array.shape[0]
label = np.load(data_path + 'label.npy')