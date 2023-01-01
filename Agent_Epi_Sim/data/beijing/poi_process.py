import numpy as np
with open('./raw_data/tencent_poi/pois_within_beijing', encoding='gbk') as f:
    f2=f.read()
f1=f2.split('\n')
f1.remove('')
poi_d={}
from tqdm import tqdm
for i in tqdm(range(len(f1))):
    if i>0:
        poi_d[f1[i].split('\t')[0]]=(eval(f1[i].split('\t')[3]),eval(f1[i].split('\t')[4]))
# poi_id:(long, lati)
np.save('./processed_data/poi_loc.npy',poi_d)