from multiprocessing import Process, Manager,Pool
import multiprocessing as mp
import numpy as np
import os
from datetime import datetime as da
import datetime as db
from tqdm import tqdm

files=os.listdir('./raw_data/beijing/')
descri_queue = Manager().Queue()

time_granularity = db.timedelta(minutes=120)

def inter_time(tt):
    # if tt.minute>30:
    tt1 =  tt - db.timedelta(minutes=tt.minute) + db.timedelta(hours=1)
    # else:
    #     tt1 =  tt - db.timedelta(minutes=tt.minute) + db.timedelta(minutes=30)
    return tt1
def one_reward(descri_queue,data_list,f2,result_queue):
    user_dict={}
    for i in data_list:
        if (i+1)%2==0 and len(eval(f2[i]))!=0:
            user_dict[f2[i-1].split('\t')[0]]={}
            t1 = da.strptime(str(eval(eval(f2[i])[0])[1]),'%Y%m%d%H%M')
            t2 = da.strptime(str(eval(eval(f2[i])[-1])[1]),'%Y%m%d%H%M')
            ori = inter_time(t1)
            last = inter_time(t2)
            num = int((last-ori).total_seconds()//(time_granularity.total_seconds()))+1
            key_list = [ori + n*time_granularity for n in range(num)]
            #for n in range(int(num)):
                #tim = ori + n*db.timedelta(minutes=30)
                #user_dict[f2[i-1].split('\t')[0]][tim] = 0
                #key_list.append(tim)
            #key_list = list(user_dict[f2[i-1].split('\t')[0]].keys())
            for j in range(len(eval(f2[i]))):
                tuu=da.strptime(str(eval(eval(f2[i])[j])[1]),'%Y%m%d%H%M')
                tu = inter_time(tuu)
                usq = int((tu-ori).total_seconds()//time_granularity.total_seconds())
                #print(key_list[usq],eval(f2[i])[j])
                user_dict[f2[i-1].split('\t')[0]][key_list[usq]]= eval(eval(f2[i])[j])[0]
    #print('successful')
    result_queue.put(user_dict)

def get_reward(idd,descri_queue,f2):
    s1=np.arange(len(f2)-len(f2)%40).reshape(40,-1).tolist()
    try:
        s1[-1].extend(np.arange(len(f2)-len(f2)%40,len(f2)).tolist())
    except:
        print(s1)
    result_queue = Manager().Queue()
    #iter_list = np.arange(1000).reshape(2,-1).tolist()
    data_dict = {}
    processes = list()
    for iu in range(40):
        p = Process(target=one_reward, args=(descri_queue,s1[iu],f2,result_queue))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for i in range(result_queue.qsize()):
        data_dict.update(result_queue.get())
    print('start to save:',idd)
    np.save('./processed_data/result_2h/1_{}.npy'.format(idd),data_dict)

# print("123313")
for pp in tqdm(range(len(files))):
    with open('./raw_data/beijing/'+files[pp]) as f:
        f1=f.read()
    print(files[pp][-2:])
    f2=f1.split('\n')
    f2.remove('')
    get_reward(files[pp][-2:],descri_queue,f2)

