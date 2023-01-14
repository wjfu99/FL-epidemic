import numpy as np
import random
from tqdm import tqdm

def init_infect(pop_info,ratio):
    for info in pop_info:
        if random.uniform(0, 1) < ratio:
            pop_info[info]['state'] = 'I'
        else:
            pop_info[info]['state'] = 'S'
    return pop_info


def gleam_epidemic(pop_info,period,ori,end):
    trace_array = []
    for info in pop_info:
        trace_array.append(pop_info[info]['trace'][ori*period*48:48*period*end])
    trace_array = np.array(trace_array)
    print(trace_array.shape)
    for j in tqdm(range(trace_array.shape[1])):
        region_infected_num = np.zeros(Region_num, dtype=int)
        region_pop_num = np.zeros(Region_num, dtype=int)
        for i in pop_info:
            if pop_info[i]['trace'][j]!=-1:
                if pop_info[i]['state'] == 'I':
                    region_infected_num[pop_info[i]['trace'][j]] += 1
                region_pop_num[pop_info[i]['trace'][j]] += 1
        region_pop_num += np.where(region_pop_num == 0, 1, 0)
        region_force = Beta * np.true_divide(region_infected_num, region_pop_num)
        for info in pop_info:
            if pop_info[info]['trace'][j]!=-1:
                if pop_info[info]['state'] == 'S':
                    if random.uniform(0, 1) < region_force[pop_info[info]['trace'][j]]:
                        pop_info[info]['state'] = 'I'
                elif pop_info[info]['state'] == 'I':
                    if random.uniform(0, 1) < Mu:
                        pop_info[info]['state'] = 'R'
    return pop_info

#应该是处理最后的尾巴数据的， 其他基本和gleam epidemic基本一样。
def on_gleam_epidemic(pop_info, ori, period):
    trace_array = []
    for info in pop_info:
        trace_array.append(pop_info[info]['trace'][ori*period*48:])
    trace_array = np.array(trace_array)
    print(trace_array.shape)
    for j in tqdm(range(trace_array.shape[1])):
        region_infected_num = np.zeros(Region_num, dtype=int)
        region_pop_num = np.zeros(Region_num, dtype=int)
        for i in pop_info:
            if pop_info[i]['trace'][j]!=-1:
                if pop_info[i]['state'] == 'I':
                    region_infected_num[pop_info[i]['trace'][j]] += 1
                region_pop_num[pop_info[i]['trace'][j]] += 1
        region_pop_num += np.where(region_pop_num == 0, 1, 0)
        region_force = Beta * np.true_divide(region_infected_num, region_pop_num)
        for info in pop_info:
            if pop_info[info]['trace'][j]!=-1:
                if pop_info[info]['state'] == 'S':
                    if random.uniform(0, 1) < region_force[pop_info[info]['trace'][j]]:
                        pop_info[info]['state'] = 'I'
                elif pop_info[info]['state'] == 'I':
                    if random.uniform(0, 1) < Mu:
                        pop_info[info]['state'] = 'R'
    return pop_info

#num是核酸检测次数，period是核算检测隔的天数。
def isolate(pop_info,num,period):
    sample_dict = {}
    new_dict = {}
    iso = {}
    period_data={}
    for i in range(num):
        sample_dict[i]={}
        iso[i] = {}
        new_info = {}
        period_data[i]={}
        pop_info = gleam_epidemic(pop_info,period,i,i+1)
        for p in pop_info:
            period_data[i][p]={}
            if pop_info[p]['state']=='S':
                period_data[i][p]['state']=0
            elif pop_info[p]['state'] == 'I':
                period_data[i][p]['state'] = 1
            else:
                period_data[i][p]['state'] = 2
            period_data[i][p]['trace']=pop_info[p]['trace'][i*period*48:48*period*(i+1)]
        sample_result = random.sample(list(pop_info.keys()),int(len(pop_info)*0.1))
        for j in sample_result:
            sample_dict[i][j]=pop_info[j]
        for j in sample_result:
            if pop_info[j]['state']!='S':
                iso[i][j]=pop_info[j]
        # 隔离核酸检测出感染的人
        for j in pop_info:
            if j not in sample_result:
                new_info[j]=pop_info[j]
            elif j in sample_result and pop_info[j]['state']=='S':
                new_info[j]=pop_info[j]
        pop_info = new_info
    pop_info = on_gleam_epidemic(pop_info,period,4)
    period_data[4]={}
    for p in pop_info:
        period_data[4][p]={}
        if pop_info[p]['state']=='S':
            period_data[4][p]['state']=0
        elif pop_info[p]['state']=='I':
            period_data[4][p]['state']=1
        else:
            period_data[4][p]['state']=2
        period_data[4][p]['trace']=pop_info[p]['trace'][4*10*48:]
    return sample_dict,pop_info,iso,period_data

def main():
    random.seed(123)
    global Region_num
    global Mu
    global Beta
    #聚合后的区域是661
    # Region_num = 661
    #聚合前区域个数
    Region_num = 3800
    parameters = 'Omicron'
    if parameters == 'Omicron':
        # Omicron
        # Mu = 0.0030639024815920513
        # Beta = 0.058656271323618725
        Mu = 0.071 / 48
        Beta = 0.766 / 48
    elif parameters == 'primitive':
        # primitive
        # Mu = 0.0029745672532136558
        # Beta = 0.019132277144042642
        # 下面的是在大尺度下的感染率
        Mu = 0.071/48
        Beta = 0.305/48
    ratio = 0.0001
    # user_info = np.load('./processed_data/ori_data_old.npy', allow_pickle=True).item()
    trajs = np.load('../beijing/processed_data/traj_mat(filled).npy')
    trajs = trajs[:, :15*48]
    user_info = {}
    for idx, traj in enumerate(trajs):
        user_info[idx] = {'trace': traj}
    # for i in range(44000):
    #     user_info.pop(random.choice(user_info.keys()))
    # for key in random.sample(user_info.keys(), 44000):
    #     del user_info[key]
    pop_info = init_infect(user_info,ratio)
    sample_dict,pop_info,iso,period_data =isolate(pop_info,1,40)
    if parameters == 'Omicron':
        np.save('./processed_data/large/sample_result_old_omicron.npy', sample_dict)
        np.save('./processed_data/large/per_data_old_omicron.npy', period_data)
    else:
        np.save('./processed_data/large/sample_result_old.npy',sample_dict)
        np.save('./processed_data/large/per_data_old.npy',period_data)

if __name__ == "__main__":
    main()
