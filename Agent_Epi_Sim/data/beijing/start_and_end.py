import os
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
import time, random
from functools import reduce
print(os.getcwd())
files = os.listdir('./processed_data/result')
# first_time=[]
# last_time=[]

def start_and_end(file_name):
    print("start process: "+file_name)
    file = np.load('./processed_data/result/'+file_name,allow_pickle=True).item()
    f = []
    l = []
    for i in file:
        f.append(list(file[i].keys())[0])
        l.append(list(file[i].keys())[-1])
        # first_time.put(list(file[i].keys())[0])
        # last_time.put(list(file[i].keys())[-1])
    print("process finished: " + file_name)
    return f, l
from tqdm import tqdm
manager = Manager()
pool = Pool(50)
# manager.dict()
first_time = manager.Queue()
last_time = manager.Queue()
f_l = pool.map(start_and_end, files)
first_time = list(zip(*f_l))[0]
# first_time = list(zip(*first_time[0]))
# x.extend() return None
first_time = reduce(lambda x, y: x.extend(y) or x, list(first_time))
last_time = list(zip(*f_l))[1]
last_time = reduce(lambda x, y: x.extend(y) or x, list(last_time))
# last_time = list(zip(*last_time[1]))

# print(f_l)
# for p in tqdm(range(len(files))):
#     pool.apply_async(start_and_end, args=(files[p],))
    # pool.apply_async(fun1, args=(p, ))
# pool.close()
# pool.join()
# print(files)
# for i in f_l:
#     print(i[2])
    # a1 = np.load('./processed_data/result/'+p,allow_pickle=True).item()
    # for i in a1:
    #     first_time.append(list(a1[i].keys())[0])
    #     last_time.append(list(a1[i].keys())[-1])
np.save('./processed_data/first.npy',first_time)
np.save('./processed_data/last.npy',last_time)