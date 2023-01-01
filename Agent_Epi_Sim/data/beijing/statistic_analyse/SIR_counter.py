import pickle as pkl
import numpy as np
from tqdm import tqdm

per_data = np.load('../processed_data/per_data_omicron.npy', allow_pickle=True).item()

label = [per_data[0][x]['state'] for x in per_data[0]]
label = np.array(label).reshape(-1, 1)
a = np.where(label==2)