import torch.nn as nn
import torch
import pdb

class Confidence(nn.Module):

    def __init__(self, label_num, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.embedding = nn.Embedding(self.label_num, self.hidden_size, -1)
        self.enc_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.enc_log_sigma = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.predict = nn.Linear(hidden_size, label_num)
        self.nll = nn.NLLLoss(ignore_index=label_num-1, reduction='none')
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        """
        Input:
        data (batch_size, seq_len)
        ---
        Output:
        p_x (batch_size, seq_len)
        """
        x = self.embedding(data)
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        q_z = torch.distributions.Normal(mu, sigma)

        z = q_z.rsample()
        p_x = self.dec(z)
        p_x = self.predict(p_x)
        p_x = self.logsoftmax(p_x)
        
        batch_size, seq_len = data.shape
        p_x = p_x.reshape(batch_size*seq_len, -1)
        data = data.reshape(-1)
        nll = self.nll(p_x, data)
        nll = nll.reshape(batch_size, seq_len)

        divergence = torch.distributions.kl_divergence(q_z, torch.distributions.Normal(0, 1))
        return nll+divergence.sum(dim=-1)