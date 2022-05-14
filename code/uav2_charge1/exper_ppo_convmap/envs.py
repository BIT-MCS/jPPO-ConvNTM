import os
from abc import abstractmethod
#
from environment_new.environment_new_uav2_charge1.env import *
from environment_new.environment_new_uav2_charge1.log import *
# from environment_uav2_charge1.env import *
# from environment_uav2_charge1.log import *
# from environment_new_new.env import *
# from environment_new_new.log import *
# import gym

# from environment_new_rand.env import *
# from environment_new_rand.log import *
import numpy as np
import torch

from util.utils import *

root_path = '/home/linc/one_storage/experiment_ppo_data/experiment3/uav2_charge1/exper_ppo_convmap/'


class Make_Env(object):
    def __init__(self, num_process, device, num_steps):
        self.num_process = num_process
        self.util = Util(device)
        self.time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
        self.env_list = [Env(Log(i, num_steps, root_path, self.time)) for i in range(num_process)]
        self.num_steps = num_steps

        self.env_action = [[] for i in range(num_process)]
        self.env_cur_energy = [[] for i in range(num_process)]

        self.env_data_collection = [[] for i in range(num_process)]
        self.env_fairness = [[] for i in range(num_process)]
        self.env_efficiency = [[] for i in range(num_process)]
        self.env_energy_consumption = [[] for i in range(num_process)]
        for i, env in enumerate(self.env_list):
            self.env_cur_energy[i].append([float(x) for x in list(env.cur_uav_energy)])
            self.env_data_collection[i].append(env.data_collection_ratio())
            self.env_fairness[i].append(env.geographical_fairness())
            self.env_efficiency[i].append(env.energy_efficiency())
            self.env_energy_consumption[i].append(env.energy_consumption_ratio())

    def reset(self):
        self.step_counter = 0
        self.totol_r = np.zeros(shape=[self.num_process], dtype=np.float32)
        obs = []
        for env in self.env_list:
            ob, _, _ = env.reset()  # [80,80,3]
            ob = ob.transpose(2, 0, 1)  # [3,80,80]
            ob = np.expand_dims(ob, axis=0)  # [1,3,80,80]
            obs.append(self.util.to_tensor(ob))
        obs = torch.cat(obs, dim=0)  # [num,3,80,80]

        return obs

    def step(self, action, current_step=None):

        self.step_counter += 1
        if self.step_counter < self.num_steps:
            done = [False for i in range(self.num_process)]
        else:
            done = [True for i in range(self.num_process)]

        obs = []
        reward = []
        info = {}

        for i, env in enumerate(self.env_list):
            # action [K,3]
            ob, r, d, _, _, _ = env.step(action[i], current_step)  # [80,80,3]
            self.env_action[i].append([float(x) for x in list(np.reshape(action[i], [-1]))])
            self.env_cur_energy[i].append([float(x) for x in list(env.cur_uav_energy)])
            self.env_data_collection[i].append(env.data_collection_ratio())
            self.env_fairness[i].append(env.geographical_fairness())
            self.env_efficiency[i].append(env.energy_efficiency())
            self.env_energy_consumption[i].append(env.energy_consumption_ratio())

            self.totol_r[i] += r

            ob = ob.transpose(2, 0, 1)  # [3,80,80]
            ob = np.expand_dims(ob, axis=0)  # [1,3,80,80]
            obs.append(self.util.to_tensor(ob))

            r = np.array([r], dtype=np.float32)  # [1]
            r = np.expand_dims(r, axis=0)  # [1,1]
            reward.append(self.util.to_tensor(r))
        obs = torch.cat(obs, dim=0)  # [num,3,80,80]
        reward = torch.cat(reward, dim=0)  # [num,1]

        info['mean_episod_reward'] = np.mean(self.totol_r)
        info['max_episod_reward'] = np.max(self.totol_r)
        info['min_episod_reward'] = np.min(self.totol_r)

        return obs, reward, done, info

    def draw_path(self, step):
        for env in self.env_list:
            env.log.draw_path(env, step)

    def test_summary(self):
        summary_txt_path = self.log_path + '/' + 'test_summary.txt'
        f = open(summary_txt_path, 'w')
        f.writelines('mean effi is : ' + str(np.mean(self.mean_efficiency)) + '\n')
        f.writelines('mean d_c is : ' + str(np.mean(self.mean_data_collection_ratio)) + '\n')
        f.writelines('mean f is : ' + str(np.mean(self.mean_fairness)) + '\n')
        f.writelines('mean e_c is : ' + str(np.mean(self.mean_energy_consumption_ratio)))
        f.close()

        summary_npz_path = self.log_path + '/' + 'test_summary.npz'
        print(self.efficiency)
        np.savez(summary_npz_path, self.efficiency, self.data_collection_ratio, self.fairness,
                 self.energy_consumption_ratio)

        self.env_uav_trace = [[[[] for _ in range(len(self.env_list[0].uav_trace))] for _ in
                               range(self.num_steps+1)] for i in range(self.num_process)]
        for i, env in enumerate(self.env_list):
            for j in range(self.num_steps+1):
                for s in range(len(env.uav_trace)):
                    if j < len(env.uav_trace[s]):
                        self.env_uav_trace[i][j][s].append([float(x) for x in list(env.uav_trace[s][j])])
                    else:
                        self.env_uav_trace[i][j][s].append([float(x) for x in list(env.uav_trace[s][len(env.uav_trace[s])-1])])
        json_dict = {}
        json_dict['action'] = self.env_action
        json_dict['trace'] = self.env_uav_trace
        json_dict['cur_energy'] = self.env_cur_energy
        json_dict['d_c'] = self.env_data_collection
        json_dict['f'] = self.env_fairness
        json_dict['e_c'] = self.env_energy_consumption
        json_dict['effi'] = self.env_efficiency

        import json

        json_str = json.dumps(json_dict)
        # print(json_str)
        # print(type(json_str))

        new_dict = json.loads(json_str)

        # print(new_dict)
        # print(type(new_dict))

        with open(self.log_path + '/' + 'record.json', "w") as f:
            json.dump(new_dict, f)

            print("加载入文件完成...")

    @property
    def log_path(self):
        return self.env_list[0].log.full_path

    @property
    def observ_shape(self):
        return self.env_list[0].observ.transpose(2, 0, 1).shape

    @property
    def action_space(self):
        return self.env_list[0].uav_num * 3

    @property
    def num_of_uav(self):
        return self.env_list[0].uav_num

    @property
    def mean_efficiency(self):
        return np.mean([env.energy_efficiency() for env in self.env_list])

    @property
    def mean_data_collection_ratio(self):
        return np.mean([env.data_collection_ratio() for env in self.env_list])

    @property
    def mean_fairness(self):
        return np.mean([env.geographical_fairness() for env in self.env_list])

    @property
    def mean_energy_consumption_ratio(self):
        return np.mean([env.energy_consumption_ratio() for env in self.env_list])

    @property
    def efficiency(self):
        return np.array([env.energy_efficiency() for env in self.env_list])

    @property
    def data_collection_ratio(self):
        return np.array([env.data_collection_ratio() for env in self.env_list])

    @property
    def fairness(self):
        return np.array([env.geographical_fairness() for env in self.env_list])

    @property
    def energy_consumption_ratio(self):
        return np.array([env.energy_consumption_ratio() for env in self.env_list])
