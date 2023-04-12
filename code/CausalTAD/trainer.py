import os
import json
import torch
import pdb
import torch.nn as nn
import numpy as np
import time

from torch.optim import SGD, Adam
from torch.nn.utils import clip_grad_value_

from .dataset import TrajectoryLoader, GraphLoader, GraphPartition
from .model import Model
from .params import Params

class Trainer:

    def __init__(self, save_model, city, cuda_devices=[1], load_model=None) -> None:
        self.params = Params(city=city)
        self.load_model = "{}_{}".format(city, load_model)
        self.cuda_devices = cuda_devices
        self.save_model = "{}_{}".format(city, save_model)
        self.city = city
        self.device = 'cuda:0'
        
        self.road_network = GraphLoader(self.params.nodes, self.params.node_adj, self.device)
        self.params.label_num = len(self.road_network.nodes) + 3

        self.model = Model(self.params.hidden_size, self.params.hidden_size, self.device, self.params.layer_rnn, self.params.label_num, len(self.road_network.edge2index))
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.params.label_num-1)
        if torch.cuda.device_count()>1 and len(cuda_devices)>1:
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        self.model = self.model.to(self.device)
        if load_model!=None:
            checkpoint = torch.load(os.path.join(self.params.save_path, "{}.pth".format(self.load_model)))
            self.model.load_state_dict(checkpoint['model'])

        self.optimizer = Adam([{"params":self.model.parameters(
                ), 'lr':self.params.lr}], weight_decay=self.params.weight_decay)


    def train_epoch(self, epoch: int, stage: int, dataloader: TrajectoryLoader):
        if stage==1 or stage==2:
            self.model.train()
            desc = "Train"
        else:
            self.model.eval()
            desc = "Test"
        
        avg_loss = 0
        order_prob = []
        tail_count = dict()

        start = time.time()
        for i, data in enumerate(dataloader.src_data_batchs):
            src, trg, src_lengths, trg_lengths = data.to(self.device), dataloader.trg_data_batchs[i].to(self.device), dataloader.src_length_batchs[i], dataloader.trg_length_batchs[i]
            sub_graph_edges = self.road_network.sample_subgraph(src)
            
            nll_loss, kl_loss, confidence, sd_loss = self.model.forward(src, trg, sub_graph_edges, src_lengths, trg_lengths)
            
            confidence_mean = confidence.mean()
            if stage==1 or stage==2:
                nll_loss = nll_loss.sum(dim=-1).mean()
                confidence = confidence.sum(dim=-1).mean()
                loss = nll_loss + kl_loss.mean() + confidence + sd_loss
                loss = loss.mean()
                avg_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                # loss = nll_loss.sum(dim=-1)/(src_lengths.to(self.device)+1)
                prob = nll_loss.cpu().detach().tolist()
                confidence_list = confidence.cpu().detach().tolist()
                src_lengths = src_lengths.cpu().detach().tolist()
                for j, item in enumerate(prob):
                    order_prob.append([src_lengths[j], item, confidence_list[j]])
                nll_loss = nll_loss.sum(dim=-1).mean()
                loss = nll_loss + kl_loss.mean()
                avg_loss += loss.mean().item()
                loss = loss.mean()
            post = "{} epoch:{}, iter:{}, avgloss:{:.4f}, nll:{:.4f}, kl:{:.4f}, conf:{:.4f}, sd_loss:{:.4f}".format(desc, epoch, i, avg_loss/(i+1), nll_loss.mean().item(), kl_loss.mean().item(), confidence_mean.item(), sd_loss.item())
                
            if i%10==0:
                print(post)
            
        
        print(1000*(time.time()-start)/(self.params.batch_size * len(dataloader.src_data_batchs)))

        with open(os.path.join(self.params.output, "log.txt"), 'a') as f:
            f.write(post + '\n')
        
        if stage==3:
            with open(os.path.join(self.params.output, "{}_prob_{}.json".format(self.load_model, epoch)), 'w') as f:
                json.dump(order_prob, f)

    def save(self, epoch):
        if torch.cuda.device_count()>1 and len(self.cuda_devices)>1:
            state = {
                'model': self.model.module.state_dict(),
                'embedding': self.model.road_embedding.module.state_dict(),
                'vae': self.model.vae.module.state_dict(),
                'projection': self.model.projection_head.module.state_dict(),
                'confidence': self.model.confidence.module.state_dict()
            }
        else:
            state = {
                'model': self.model.state_dict(),
                'embedding': self.model.road_embedding.state_dict(),
                'vae': self.model.vae.state_dict(),
                'projection': self.model.projection_head,
                'confidence': self.model.confidence.state_dict()
            }
        torch.save(state, os.path.join(self.params.save_path, "{}_{}.pth".format(self.save_model, epoch)))



    def train(self):
        self.train_dataset = TrajectoryLoader(self.params.train_dataset, self.road_network.node2id, self.params.batch_size, self.params.label_num)
        for i in range(self.params.epochs):
            self.train_epoch(i, 1, self.train_dataset)
            # if i%10==0:
            self.save(i)

    def test(self):
        self.params.batch_size = 64
        with torch.no_grad():
            self.normal_dataset = TrajectoryLoader(self.params.normal_dataset, self.road_network.node2id, self.params.batch_size, self.params.label_num)
            self.train_epoch(0, 3, self.normal_dataset)
            self.detour_dataset = TrajectoryLoader(self.params.detour_dataset, self.road_network.node2id, self.params.batch_size, self.params.label_num)
            self.train_epoch(1, 3, self.detour_dataset)
            self.switch_dataset = TrajectoryLoader(self.params.switch_dataset, self.road_network.node2id, self.params.batch_size, self.params.label_num)
            self.train_epoch(2, 3, self.switch_dataset)
            self.ood_dataset = TrajectoryLoader(self.params.ood_dataset, self.road_network.node2id, self.params.batch_size, self.params.label_num)
            self.train_epoch(3, 3, self.ood_dataset)
            # self.case_dataset = TrajectoryLoader(self.params.ood_detour, self.road_network.node2id, self.params.batch_size, self.params.label_num)
            # self.train_epoch(4, 3, self.case_dataset)