import numpy as np
from tqdm import tqdm

def location_count(usd):
    location={}
    for i in tqdm(usd):
        for j in usd[i]:
            if usd[i][j] not in location:
                location[usd[i][j]]=1
            else:
                location[usd[i][j]]+=1
    np.save('./processed_data/bj_loc.npy',location)
    return location

def sort_and_count(location,loc_d):
    max_traj=sorted(location,key=lambda x:location[x])
    max_count = max_traj[-1]
    all_poi={}
    u=[]
    for i in location:
        try:
            all_poi[i]=loc_d[str(i)]
        except:
            #print(i)
            u.append(i)
    return all_poi


def filter_and_map(usd,all_poi):
    apa={}
    for us in tqdm(usd):
        c=0
        for i in usd[us]:
            if usd[us][i] in all_poi:
                c = c
            else:
                c = c+1
        if c==0: #如果用户存在poi集合里面找不到的poi那么就删掉这条轨迹。
            apa[us]=usd[us]
    return apa


def region_filter(apa, all_poi, usd):
    lon=[]
    for i in tqdm(apa):
        c=0
        for j in apa[i]:
            if 115.838<all_poi[apa[i][j]][0]<116.4969 and 39.88963<=all_poi[apa[i][j]][1]<40.3869:
                c=c
            else:
                c+=1
        if c==0:
            lon.append(i)
    print(len(set(lon)))
    apa1={}
    for i in lon:
        apa1[i]=usd[i]
    np.save('./processed_data/final_user.npy',apa1)

def main():
    print("------load data-------")
    usd = np.load('./processed_data/userd.npy', allow_pickle=True).item()
    loc_d = np.load('./processed_data/poi_loc.npy', allow_pickle=True).item()
    print('-----step 1-----------')
    location = location_count(usd)
    print('-----step 2-----------')
    all_poi = sort_and_count(location,loc_d)
    print('-----step 3-----------')
    apa = filter_and_map(usd,all_poi)
    print('-----process and save-----------')
    region_filter(apa, all_poi,usd)

if __name__ == "__main__":
    main()


