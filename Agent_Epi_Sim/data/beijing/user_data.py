import os
import numpy as np
from datetime import datetime as da
import datetime as db
from tqdm import tqdm
first=np.load('./processed_data/first.npy',allow_pickle=True)
first_set=set(first)
first_list=first.tolist()
uu=[]
for i in tqdm(first_set):
    uu.append((i,first_list.count(i)))
max_first=sorted(uu,key=lambda x:x[1])
last=np.load('./processed_data/last.npy',allow_pickle=True)
last_set=set(last)
last_list=last.tolist()
uu1=[]
#from tqdm import tqdm
for i in tqdm(last_set):
    uu1.append((i,last_list.count(i)))
max_last=sorted(uu1,key=lambda x:x[1])
start_time = max_first[-1][0]
end_time = max_last[-1][0]-db.timedelta(hours=max_last[-1][0].hour)-db.timedelta(minutes = max_last[-1][0].minute)
time_bin = int((end_time-max_first[-1][0]).total_seconds()/1800)+1

user_dict={}
files = os.listdir('./processed_data/result')
#percent=[]
for p in tqdm(files):
    a1 = np.load('./processed_data/result/'+p,allow_pickle=True).item()
    for u in a1:
        bin_list=[]
        for j in a1[u]:
            if j>=start_time and j<=end_time:
                bin_list.append(j)
        if len(bin_list)/time_bin>0.1:
            user_dict[u]=a1[u]
np.save('./processed_data/userd.npy',user_dict)