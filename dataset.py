#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020-10-10 23:15
# @Author : Kun Wang(Adam Dylan)
# @File : model.py.py
# @Comment : Created By Kun Wang,23:15
# @Completed : Yes
# @Tested : Yes

import numpy as np
from sklearn.preprocessing import *
from torch.utils.data import Dataset
import torch as t

class TimeSeriesDataset(Dataset):

    def minmaxscale(self, data: np.ndarray):
        seq_len, num_features = data.shape
        for i in range(num_features):
            min = data[:, i].min()
            max = data[:, i].max()
            data[:, i] = (data[:, i] - min) / (max - min)
        return data, min, max

    def __init__(self, params, scaler=StandardScaler()):
        import pandas as pd

        self.encode_step = params['encode_step']
        self.forcast_step = params['forcast_step']
        rawdata = pd.read_csv('dataset.csv')

    # 1.带0预测（label补足）       
        self.features = rawdata.fillna(0)  
        self.features = self.features.to_numpy().astype('float32')
        self.features = self.features[:, 0:4]
        self.features, self.xmin , self.xmax = self.minmaxscale(self.features)

        self.label = rawdata.fillna(method = 'ffill', inplace=False)           
        self.label = self.label.to_numpy().astype('float32')
        self.label = self.label[:, 4]
        self.label,self.zmin,self.zmax = self.minmaxscale(self.label.reshape(-1,1))


        
    # 2.不带0预测（全数据补足） 
        # target = rawdata.fillna(method = 'ffill', inplace=False)   
        # target = target.to_numpy().astype('float32')
        # target = self.minmaxscale(target)

        # features = rawdata.fillna(method = 'ffill', inplace=False)  
        # features = features.to_numpy().astype('float32')
        # features = self.minmaxscale(features)

        # self.featuress = rawdata[:, 0:4]
        # self.label = testdata[:, 4]
        

        self.features, _, _ = self.minmaxscale(self.features)
        self.scaler = scaler
        self.scaler.fit(self.label.reshape(-1, 1))
        self.label = scaler.transform(self.label.reshape(-1, 1)).astype('float32')

    def __getitem__(self, index):
        # features
        # [index + 1, t0] [t0 + 1, T]
        # lag input
        # [index, t0 - 1]
        
        #rowfea = self.featuress[index]
        
        
        # Step 1: lagged size adjust
        index += 1

        # Step 2: history featuress
        start = index
        end = start + self.encode_step
        hisx = self.features[start:end]

        # Step 3: history inputs
        hisz = self.label[start - 1:end - 1]

        # Step 4: future featuress
        start = end + 1
        end = start + self.forcast_step
        futx = self.features[start:end]

        # Step 5: targets
        z = self.label[index: index + self.encode_step + self.forcast_step]

        # ha = np.mean(self.label[0:index + self.encode_step])
        # HA = []
        # for i in range(12):
        #         HA.append(np.float32(ha))
        # HA = np.array(HA)
        return hisx, hisz, futx, z#, HA

    def __len__(self):
        return len(self.features) - self.forcast_step - self.encode_step - 1
