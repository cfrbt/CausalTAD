from cmath import isnan
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sp_matmul, sp_softmax

class SPGNNLayers(nn.Module):

    def __init__(self, hidden_size: int, edge_num: int, node_num: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_weight = nn.Parameter(torch.ones(edge_num))
        self.node_num = node_num

    def forward(self, x, edge_list, edge2id):
        """
        x (hidden_size, label_num): the projection head
        edge_list (2, edge_num): the sampled edges
        edge2id (edge_num): the index of sampled edges
        """
        # (edge_num)
        edge_weight = sp_softmax(edge_list, self.edge_weight[edge2id], self.node_num)
        x = sp_matmul(edge_list, edge_weight, x.t())
        return x.t()