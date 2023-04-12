import pdb
from .GNNLayer import SPGNNLayers

import torch
import torch.nn as nn
import torch.nn.functional as F

class SPGNN(nn.Module):
    def __init__(self, edge_num: int, nhid: int, node_num: int) -> None:
        super().__init__()
        self.gnn = SPGNNLayers(hidden_size=nhid, edge_num=edge_num, node_num=node_num)
    
    def forward(self, projection_head, sub_edge, edge2index):
        projection_head = self.gnn(projection_head, sub_edge, edge2index)
        return projection_head