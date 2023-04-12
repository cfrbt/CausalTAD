from copy import deepcopy
import pickle
import pdb
from random import random, shuffle
import torch
import numpy as np

class TrajectoryLoader:

    def __init__(self, trajectory_path: str, node2id: dict, batch_size: int, label_num: int, shuffle: bool=True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.label_num = label_num
        self.shuffle = shuffle
        self.load_data(trajectory_path, node2id)
        self.bos_eos_pad()
        self.batch_preprocess()

    def load_data(self, trajectory_path, node2id):
        dataset = pickle.load(open(trajectory_path, 'rb'))
        self.data = []
        for line in dataset.values():
            traj = line['n_geo']
            item = []
            for node in traj:
                item.append(node2id[str(node)])
            self.data.append(item)
        if self.shuffle:
            shuffle(self.data)

    def bos_eos_pad(self):
        self.bos = self.label_num-3
        self.eos = self.label_num-2
        self.pad = self.label_num-1

    def batch_preprocess(self):

        self.src_data_batchs = []
        self.trg_data_batchs = []
        self.src_length_batchs = []
        self.trg_length_batchs = []

        for i in range(0, len(self.data), self.batch_size):
            if i+self.batch_size>len(self.data):
                cur_batch = self.data[i:len(self.data)]
            else:
                cur_batch = self.data[i:i+self.batch_size]
            
            src_length = []
            trg_length = []
            trg_batch = []
            src_batch = []

            for item in cur_batch:
                # item: (level_num, seq_len)
                src_length.append(len(item))
                trg_batch.append([self.bos] + deepcopy(item) + [self.eos])
                trg_length.append(len(trg_batch[-1]))
                src_batch.append(item)
                
            max_length = max(src_length)

            for item in src_batch:
                item += [self.pad]*(max_length-len(item))
            
            for item in trg_batch:
                item += [self.pad]*(max_length+2-len(item))
            
            self.src_data_batchs.append(torch.LongTensor(src_batch))
            self.trg_data_batchs.append(torch.LongTensor(trg_batch))
            self.src_length_batchs.append(torch.IntTensor(src_length))
            self.trg_length_batchs.append(torch.IntTensor(trg_length))