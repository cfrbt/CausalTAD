import pdb
import torch
import math
import torch.nn as nn
import numpy as np
from .GAT import SPGNN
from .vae import VAE
from .confidence import Confidence

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, device, layer_rnn, label_num, edge_num) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.confidence = Confidence(label_num, hidden_size)
        self.vae = VAE(input_size, hidden_size, layer_rnn, hidden_size, 0, label_num)
        self.road_embedding = nn.Embedding(self.label_num, hidden_size)
        self.projection_head = nn.Parameter(torch.randn(hidden_size, label_num))
        self.sd_projection_head = nn.Parameter(torch.randn(hidden_size, label_num))
        self.gnn = SPGNN(edge_num, hidden_size, self.label_num)
        self.sd_loss = nn.NLLLoss()
        self.log_soft = nn.LogSoftmax(dim=-1)

    def loss_fn(self, p_x, target, mask):
        """
        Input:
        p_x (batch_size*seq_len, hidden_size): P(target|z)
        target (batch_size*seq_len) : the target sequences
        mask (batch_size*seq_len, vocab_size): the mask according to the road network
        """
        # masked softmax
        p_x = torch.exp(p_x)
        p_x = p_x*mask.float()
        masked_sums = p_x.sum(dim=-1, keepdim=True) + 1e-6
        p_x = p_x/masked_sums
        p_x[:, self.label_num-1] = 1

        p_x = p_x[torch.arange(target.size(0)).to(target.device), target]
        nll = -torch.log(p_x)
        
        return nll

    def get_mask(self, edge_list, label, batch_size, seq_len):
        source, target = edge_list
        # (batch_size*seq_len, sub_edge_num)
        source = source.unsqueeze(0).repeat(label.shape[0], 1)
        target = target.unsqueeze(0).repeat(label.shape[0], 1)
        # (batch_size*seq_len, sub_edge_num)
        source = (source==(label.unsqueeze(1).repeat(1, edge_list.shape[1]))).long()
        # (batch_size*seq_len, label_num)
        mask = torch.zeros(label.shape[0], self.label_num).long().to(source.device)
        # mask[i][target[i][j]] = src[i][j]
        mask.scatter_add_(dim=1, index=target, src=source)
        # (batch_size*seq_len, node_num) => (batch_size, seq_len, node_num)
        mask = mask.view(batch_size, seq_len, -1).contiguous()
        mask = torch.cat((torch.ones(batch_size, 1, self.label_num).to(mask.device), mask[:, :-1, :]), dim=1)
        mask[:, :, self.label_num-2] = 1
        mask = mask.view(batch_size*seq_len, -1).contiguous()
        return mask

    def forward(self, src, trg, edge_list, src_lengths=None, trg_lengths=None):
        """
        Input:
        src (batch_size, seq_len): the input sequence
        trg (batch_size, seq_len): the target sequence
        edge_list (2, edge_num): edges in the selected subgraph
        stage (int): indicates the first stage or the second stage
        src_length (batch_size): lengths of input sequences
        trg_length (batch_size): lengths of target sequences
        ---
        Output:
        loss (batch_size): loss
        """
        confidence = self.confidence(src)

        batch_size, seq_len = src.size(0), src.size(1)+1
        cond_trg = src[torch.arange(batch_size), (src_lengths-1).long()].unsqueeze(1)
        cond_src = src[:, 0].unsqueeze(1)
        src = torch.cat((cond_src, cond_trg), dim=-1)
        sd = src
        src = self.road_embedding(src)
        src_lengths = torch.zeros([batch_size]).long() + 2

        label = torch.clone(trg[:, 1:])
        trg = self.road_embedding(trg)
        kl_loss, p_x, sd_p_x = self.vae.forward(src, trg, src_lengths, trg_lengths)

        # (batch_size, seq_len, hidden_size) => (batch_size*seq_len, )
        p_x = p_x.view(batch_size*seq_len, -1)
        p_x = p_x.mm(self.projection_head)
        label = label.reshape(-1)
        mask = self.get_mask(edge_list, label, batch_size, seq_len)
        nll_loss = self.loss_fn(p_x, label, mask)
        nll_loss = nll_loss.view(batch_size, seq_len)
        
        sd_p_x = sd_p_x.view(batch_size*2, -1)
        sd_p_x = sd_p_x.mm(self.sd_projection_head)
        sd_p_x = self.log_soft(sd_p_x)
        sd = sd.view(-1)
        sd_nll_loss = 0.1*self.sd_loss(sd_p_x, sd)

        
        return nll_loss, kl_loss, confidence, sd_nll_loss
