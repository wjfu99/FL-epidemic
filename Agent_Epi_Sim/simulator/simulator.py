import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import random, datetime
# from time import date
from collections import Counter


class Individual:
    def __init__(self, uid, traj):
        self.residence = None
        self.state = None
        self.traj = traj
        self.position = None
        self.uid = uid
        self.info = []

    def eval_home(self):
        # for i in range(self.traj.shape[0]):
        locs, time = np.unique(self.traj, return_counts=True)
        freq = dict(zip(locs, time))
        # for debug
        self.info.append(freq)
        freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)
        self.residence = freq[1][1]
        # asure valid residence
        assert self.residence != -1


    def set_individual_isolate_days(self, days):
        """
        can not contact to acquaintance or residential
        """
        pass
    
    @property
    def get_state(self):
        return self.state

class Location:
    def __init__(self, lid):

        self.lid = lid
        self.info = None
        self.cur_usr = {}
        self.resident = {}
        # self.infected_rate = None
        # self.state_cnt = None
        # self.loc_force = None
        # self.

    @property
    def get_cur_count(self):
        return len(self.cur_usr)

    @property
    def get_state_count(self):
        # for usr in self.cur_usr.values():
        state_list = [usr.state for usr in self.cur_usr.values()]
        state_cnt = Counter(state_list)
        return state_cnt

    @property
    def get_loc_force(self):
        if self.lid != -1:
            state_cnt = self.get_state_count
            return state_cnt['I']/state_cnt['S']
        else:
            return 0

    def add_visitor(self, uid, usr):
        self.cur_usr.update({uid: usr})

class Engine:
    def __init__(self, **kwargs):
        epi_para = ['beta', 'eps', 'mu']
        # set attributes
        for key, value in kwargs.items():
            if key in epi_para:
                value = self.parameter_convert(value, time_granularity=kwargs['time_granularity'])
            setattr(self, key, value)
        # self.mu = mu
        # self.beta = beta
        self.usr_dic = {}
        self.loc_dic = {}
        self.traj_mat = None
        self.time_indi = 0

    def refresh(self):
        for loc in self.loc_dic:
            loc = self.loc_dic[loc]
            loc.cur_usr = {}

        for usr in self.usr_dic:
            usr = self.usr_dic[usr]
            # load user current location
            loc_id = usr.traj[self.time_indi]
            # TODO: the address trajectory sparsity
            # if loc_id != -1:
            usr.position = self.loc_dic[loc_id]
            usr.position.add_visitor(usr.uid, usr)
            # else:
            #     usr.position = None
            # load location current users

        # assess the infected para for each location.
        # for loc in self.loc_dic:
        #     loc = self.loc_dic[loc]
        #     loc.state_cnt = loc.get_state_count


    def add_usr(self, uid, traj):
        usr = Individual(uid, traj)
        # usr.traj = traj
        usr.eval_home()
        self.usr_dic.update({uid: usr})

    def add_loc(self, lid):
        usr = Location(lid)
        self.loc_dic.update({lid: usr})

    # def add_loc(self, loc):
    #     self.loc_dic.update(loc)

    def load_traj(self, traj: np.ndarray, specified_id=True):
        self.traj_mat = traj
        usr_num = traj.shape[0]
        # for extension
        if specified_id:
            for i in range(usr_num):
                self.add_usr(i, traj[i, :])
            locs = np.unique(traj)
            for loc in locs:
                # if loc != -1:
                self.add_loc(lid=loc)

    def init_epi(self, init_mode, ratio, **kwargs):
        if init_mode == "global":
            for usr in self.usr_dic:
                usr = self.usr_dic[usr]
                if random.uniform(0, 1) < ratio:
                    usr.state = 'I'
                else:
                    usr.state = 'S'
        elif init_mode == "poi":
            if kwargs.specified_pois:
                for usr in self.usr_dic:
                    usr = self.usr_dic[usr]
                    if usr.residence in kwargs.specified_pois:
                        if random.uniform(0, 1) < ratio:
                            usr.state = 'I'
                        else:
                            usr.state = 'S'
            elif kwargs.random_pois:
                raise

    def get_infected_sample(self):
        """
        get infected sample, for model training/isolated
        :return:
        """

    def next(self, step_num=1, dynamic_mode="poi_shared"):
        for i in range(step_num):
            self.refresh()
            if dynamic_mode == "poi_shared":
                for usr in self.usr_dic:
                    usr = self.usr_dic[usr]
                    if usr.state == 'S':
                        if random.uniform(0, 1) < self.beta * usr.position.get_loc_force*1000000:
                            usr.state = 'E'
                    elif usr.state == 'E':
                        if random.uniform(0, 1) < self.eps:
                            usr.state = 'I'
                    elif usr.state == 'I':
                        if random.uniform(0, 1) < self.mu:
                            usr.state = 'R'
            self.time_indi += 1

    @staticmethod
    def parameter_convert(para, time_granularity: datetime.timedelta, default_granularity=datetime.timedelta(days=1)):
        scale = time_granularity/default_granularity
        # for arg in args:
        #     arg /= scale
        # return [arg/scale for arg in args]
        return para * scale

    @property
    def get_current_time(self):
        return self.time_indi

    @property
    def get_state_count(self):
        # for usr in self.cur_usr.values():
        total_cnt = {}
        for loc in self.loc_dic.values():
            loc_cnt = loc.get_state_count
            total_cnt = Counter(total_cnt) + Counter(loc_cnt)
            # total_cnt = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
        return total_cnt
    
    @property
    def get_traj_mat(self):
        return self.traj_mat

    @property
    def get_usr_num(self):
        return len(self.usr_dic)
    
    @property
    def get_usr_states(self):
        # class_dic = {
        #     'S':0,
        #     'E':1,
        #     'I':2,
        #     'R':3
        # }
        label=[]
        for usr in self.usr_dic.values():
            state = usr.get_state
            # state = class_dic.get(state)
            label.append(state)
        # lbls = np.array(lbls)
        return label
    
    
    