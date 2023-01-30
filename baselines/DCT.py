import numpy as np
from tqdm import tqdm

# trace_array = np.load('./privacy/noposterior/trace_array.npy')
# # trace_array1 = np.load('./obfuscate_data/ori_data_eps=200.npy', allow_pickle=True).item()
# label = np.load('./privacy/label_omicron.npy')
# # label = np.load('./privacy/label.npy')
# x = trace_array.max()

data_path = '../datasets/beijing/large-filled-clustered/'
trace_array = np.load(data_path + "traj_mat(filled,sample).npy")
trace_array = trace_array[:, 14:48]
pop_num = trace_array.shape[0]
label = np.load(data_path + 'label_omicron.npy')


confirmed_ratio = 0.4
confirmed_num = int(pop_num * confirmed_ratio)

test = list(range(confirmed_num))
test_label = label[test]
confrim = np.argwhere(test_label == 1)[:, 0]

val = list(range(confirmed_num, pop_num))
val_label = label[val]
val_trace = trace_array[val]

contacts = []

for c in tqdm(confrim):
    trace = trace_array[c]
    trace[trace==-1] = 20000
    a = np.equal(val_trace, trace).sum(axis=1)
    a = np.argwhere(a != 0)[:, 0]
    a = a
    contacts.extend(list(a))
contacts = list(set(contacts))

tmp = val_label[contacts]
TP = tmp.sum()
FP = len(contacts) - TP
FN = val_label.sum() -TP
TN = (len(val_label)-val_label.sum()) - FP

precision = TP/len(contacts)
recall = TP/val_label.sum()
F1 = 2*precision*recall/(precision+recall)

acc = (TP + TN)/(TP+FP+FN+TN)