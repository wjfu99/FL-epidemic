import numpy as np


def label_generator(state_list):
    class_dic = {
        'S': 0,
        'E': 1,
        'I': 1,
        'R': 1
    }
    label = [class_dic[state] for state in state_list]
    label = np.array(label)
    return label
