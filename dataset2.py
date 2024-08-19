import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class TrainSet(Dataset):
    """城市A全部作为训练集"""
    def __init__(self, path):

        # 初始化
        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []
        self.city_array = []
        
        # 读取数据
        traj_df = pd.read_csv(path, compression='gzip')

        for _, traj in tqdm(traj_df.groupby('uid')):

            # 全部转换为numpy数组
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())   # 创建独立副本
            input_y = copy.deepcopy(traj['y'].to_numpy())   # 创建独立副本
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                    (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()

            # 训练时，对每个uid分组随机mask掉长度为15天的连续序列
            d_unique = np.unique(d)
            if len(d_unique[(d_unique >= np.min(d_unique)) & 
                            (d_unique <= np.max(d_unique) - 14)]) == 0:
                continue
            mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & 
                                                        (d_unique <= np.max(d_unique) - 14)])
            mask_d_end = mask_d_start + 14
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            # 将所有uid分组整合到一起并保存
            self.d_array.append(d + 1)  # 1-75; 0:<pad>
            self.t_array.append(t + 1)  # 1-48; 0:<pad>
            self.input_x_array.append(input_x)  # 1-200; 0:<pad>; 201:<mask>
            self.input_y_array.append(input_y)  # 1-200; 0:<pad>; 201:<mask>
            self.time_delta_array.append(time_delta)    # 0-47; 0:<pad>
            self.label_x_array.append(label_x - 1)  # 0-199
            self.label_y_array.append(label_y - 1)  # 0-199
            self.len_array.append(len(d))   # 每个uid分组（即每条用户轨迹）的长度
    
        self.len_array = np.array(self.len_array, dtype=np.int64)   # 转换为numpy数组

    def __len__(self):
        return len(self.d_array)
    
    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        city = torch.tensor(self.city_array[index]) # add
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])
        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'city': city,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }
    

class FineTuneSet(Dataset):
    def __init__(self, path):

        # 初始化
        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []
        self.city_array = []

        #读取数据
        traj_df = pd.read_csv(path, compression='gzip')
        for _, traj in tqdm(traj_df.groupby('uid')):

            # 全部转换为numpy数组
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())   # 创建独立副本
            input_y = copy.deepcopy(traj['y'].to_numpy())   # 创建独立副本
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                    (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()

            # 训练时，对每个uid分组随机mask掉长度为15天的连续序列
            d_unique = np.unique(d)
            if len(d_unique[(d_unique >= np.min(d_unique)) & 
                            (d_unique <= np.max(d_unique) - 14)]) == 0:
                continue
            mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & 
                                                        (d_unique <= np.max(d_unique) - 14)])
            mask_d_end = mask_d_start + 14
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            # 将所有uid分组整合到一起并保存
            self.d_array.append(d + 1)  # 1-75; 0:<pad>
            self.t_array.append(t + 1)  # 1-48; 0:<pad>
            self.input_x_array.append(input_x)  # 1-200; 0:<pad>; 201:<mask>
            self.input_y_array.append(input_y)  # 1-200; 0:<pad>; 201:<mask>
            self.time_delta_array.append(time_delta)    # 0-47; 0:<pad>
            self.label_x_array.append(label_x - 1)  # 0-199
            self.label_y_array.append(label_y - 1)  # 0-199
            self.len_array.append(len(d))   # 每个uid分组（即每条用户轨迹）的长度
        
        self.len_array = np.array(self.len_array, dtype=np.int64)   # 转换为numpy数组

    def __len__(self):
        return len(self.d_array)
    
    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        city = torch.tensor(self.city_array[index]) # add
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])
        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'city': city,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }
    