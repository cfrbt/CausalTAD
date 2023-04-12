import pdb
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DecoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, layer_num, latent_num, dropout, label_num) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_num = layer_num
        self.hidden_linear = nn.Linear(latent_num, hidden_size*layer_num)
        self.lstm = nn.GRU(input_size, hidden_size, layer_num, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(self.dropout)
        self.label_num = label_num

    def forward(self, z, target, lengths=None, train=True):
        """
        Input:
        z (batch_size, latent_size): the latent variable
        target (batch_size, seq_len, hidden_size): padded sequence tensor
        lengths (batch_size): lengths of the target sequences
        train (bool)
        ---
        Output:
        p_x (batch, seq_len, hidden_size)
        """
        hidden = self.hidden_linear(z)
        hidden = hidden.view(hidden.size(0), self.layer_num, self.hidden_size).transpose(0, 1).contiguous()
        if train:
            if lengths is not None:
                packed_input = pack_padded_sequence(target, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.lstm(packed_input, hidden)

            if lengths is not None:
                output = pad_packed_sequence(output, batch_first=True)[0]

            p_x = self.dropout(output)

        else:
            outputs = []
            target_len = target.shape[1]
            for i in range(target_len):
                # (batch_size, 1, hidden_size)
                if i==0:
                    e = target[:, 0, :].unsqueeze(1)
                else:
                    e = outputs[-1].unsqueeze(1)
                # output: (batch_size, 1, hidden_size)
                output, hidden = self.lstm(e, hidden)
                output = output.squeeze(1)
                output = self.dropout(output)
                outputs.append(output)
            outputs = torch.stack(outputs)
            outputs = self.dropout(outputs)
            p_x = outputs.transpose(0, 1).contiguous()

        return p_x
