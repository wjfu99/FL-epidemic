import datetime
import numpy as np
import datetime as db
from datetime import datetime as da
from tqdm import tqdm
import copy

def count_first_and_last(user_traj):
    first=[]
    last=[]
    for i in user_traj:
        key_list = list(user_traj[i].keys())
        first.append(key_list[0])
        last.append(key_list[-1])
    return first, last


def get_max(arr):
    last_set=set(arr)
    uu1=[]
    #from tqdm import tqdm
    for i in tqdm(last_set):
        uu1.append((i,arr.count(i)))
    max_last=sorted(uu1,key=lambda x:x[1])
    return max_last[-1][0]

# 填充轨迹，轨迹缺失部分用-1来替代
# No longer fill vacant trajectory points.
def map_user_dict(start_time, end_time, first, last, user_traj):
    user_dict = {}
    for idx, u in enumerate(tqdm(user_traj)):
        if first[idx] < start_time and last[idx] > end_time:
            user_dict[u] = user_traj[u]
            for time in list(user_dict[u].keys()):
                if time < start_time or time > end_time:
                    user_dict[u].pop(time)
    return user_dict

def id_map(user_dict):
    user_id = {}
    all_dict = {}
    for i, j in enumerate(user_dict):
        all_dict[i] = user_dict[j]
        user_id[i] = j
    np.save('./processed_data/user_id(filled).npy', user_id)
    return user_id, all_dict


def traj_count_poi(all_dict):
    locations = set()
    for i in tqdm(all_dict):
        traj = set(all_dict[i].values())
        locations = locations | traj
    loc_list = list(locations)
    trace_array = []
    for info in all_dict:
        trace_array.append(len(all_dict[info]))
    poi_id = {}
    for i in range(len(loc_list)):
        poi_id[loc_list[i]] = i
    return locations, trace_array, poi_id

def traj_covert(a2d):
    trace_array = []
    for info in a2d:
        trace_array.append(a2d[info]['trace'])
    trace_array = np.array(trace_array)
    return trace_array
# TODO: 需要处理某些用户访问的轨迹在POI信息列表里找不到的问题。
def data_process_save(all_dict, poi_id):
    a1_d = {}
    for u in tqdm(all_dict):
        a1_d[u] = {}
        for j in all_dict[u]:
            assert all_dict[u][j] != -1
            if str(all_dict[u][j]) in poi_id:
                a1_d[u][j] = int(poi_id[str(all_dict[u][j])])
            else:
                nopoi.add(all_dict[u][j])
    a2d = {}
    for i in a1_d:
        a2d[i] = {}
        a2d[i]['trace'] = list(a1_d[i].values())
    traj_mat = traj_covert(a2d)
    np.save('./processed_data/ori_data(filled).npy', a2d)
    np.save('./processed_data/traj_mat(filled).npy', traj_mat)

nopoi = set()
def main():
    # 这里导入的轨迹已经是poi的id了
    user_traj = np.load('./processed_data/filled_trajs.npy', allow_pickle=True).item()
    first, last = count_first_and_last(user_traj)
    # Manually select strat_time and end_time.
    start_time = datetime.datetime(year=2016, month=9, day=19, hour=0)
    end_time = datetime.datetime(year=2016, month=10, day=24, hour=0)
    # Select trajectories in the selected time boundary.
    user_dict = map_user_dict(start_time, end_time, first, last, user_traj)
    # user_dict还是以usrid的这里重新划分id
    user_id, all_dict = id_map(user_dict)
    # locations所有poi_id的set poi_id原始poiid对应新分的id
    locations, trace_array, poi_id = traj_count_poi(all_dict)
    # 新增的poi_区域聚合之后的id
    poi_id = np.load("./processed_data/poi_loc2region(wgs84).npy", allow_pickle=True).item()
    data_process_save(all_dict, poi_id)


if __name__ == "__main__":
    main()







