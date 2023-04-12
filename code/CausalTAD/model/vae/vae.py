import pdb
from random import random

import torch
import torch.nn as nn

from .encoder import EncoderRNN
from .decoder import DecoderRNN
from .sd_decoder import DecoderSD

class VAE(nn.Module):
    
    def __init__(self, input_size, hidden_size, layer_num, latent_num, dropout, label_num) -> None:
        super().__init__()
        self.enc = EncoderRNN(input_size, hidden_size, layer_num, latent_num, dropout, True)
        self.dec = DecoderRNN(input_size, hidden_size, layer_num, latent_num, dropout, label_num)
        self.sd_dec = DecoderSD(hidden_size, latent_num)
        self.nll = nn.NLLLoss(ignore_index=label_num-1, reduction='none')
        self.label_num = label_num

    def forward(self, src, trg, src_lengths=None, trg_lengths=None):
        """
        Input:
        src (batch_size, seq_len, hidden_size): input sequence tensor
        trg (batch_size, seq_len, hidden_size): the target sequence tensor
        src_length (batch_size): lengths of input sequences
        trg_length (batch_size): lengths of target sequences
        ---
        Output:
        kl_loss
        p_x (batch_size, seq_len)
        """
        q_z, mu, sigma = self.enc.forward(src, src_lengths)
        # (batch_size, latent_size)
        z = q_z.rsample()
        # (batch_size, seq_len, hidden_size)
        p_x = self.dec.forward(z, trg[:, :-1], trg_lengths-1)
        kl_loss = torch.distributions.kl_divergence(q_z, torch.distributions.Normal(0, 1.)).sum(dim=-1)
        sd_p_x = self.sd_dec.forward(z)
        return kl_loss, p_x, sd_p_x