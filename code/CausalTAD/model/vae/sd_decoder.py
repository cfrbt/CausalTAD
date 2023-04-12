import pdb
import torch.nn as nn


class DecoderSD(nn.Module):
    
    def __init__(self, hidden_size, latent_num) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_linear = nn.Linear(latent_num, hidden_size*2)

    def forward(self, z):
        """
        Input:
        z (batch_size, latent_size): the latent variable
        ---
        Output:
        hidden (batch, 2, hidden_size)
        """
        hidden = self.hidden_linear(z)
        hidden = hidden.view(hidden.size(0), 2, self.hidden_size).contiguous()
        
        return hidden
