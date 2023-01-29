import pickle as pkl
import numpy as np
from tqdm import tqdm


sample_result = np.load('../Agent_Epi_Sim/data/beijing/processed_data/large/sample_result_old_omicron.npy', allow_pickle=True).item()
per_data = np.load('../Agent_Epi_Sim/data/beijing/processed_data/large/per_data_old_omicron.npy', allow_pickle=True).item()

# sample_result = np.load('../Agent_Epi_Sim/data/beijing/processed_data/large/sample_result_old.npy', allow_pickle=True).item()
# per_data = np.load('../Agent_Epi_Sim/data/beijing/processed_data/large/per_data_old.npy', allow_pickle=True).item()
label = [per_data[0][x]['state'] for x in per_data[0]]
label = np.array(label).reshape(-1, 1)
label = np.where(label==2, 1, label)
label1 = [sample_result[0][x]['state'] for x in sample_result[0]]
label1 = np.array(label1).reshape(-1, 1)
np.save("./label.npy", label)
print(label.sum())