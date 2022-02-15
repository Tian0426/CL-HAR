import torch
import torch.nn as nn
import numpy as np
from .attention import _Seq_Transformer


class TC(nn.Module):
    def __init__(self, bb_dim, device, tc_hidden=100, temp_unit='tsfm'):
        super(TC, self).__init__()
        self.num_channels = bb_dim
        self.timestep = 6
        self.Wk = nn.ModuleList([nn.Linear(tc_hidden, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        self.temp_unit = temp_unit
        if self.temp_unit == 'tsfm':
            self.seq_transformer = _Seq_Transformer(patch_size=self.num_channels, dim=tc_hidden, depth=4, heads=4, mlp_dim=64)
        elif self.temp_unit == 'lstm':
            self.lstm = nn.LSTM(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                                batch_first=True, bidirectional=False)
        elif self.temp_unit == 'blstm':
            self.blstm = nn.LSTM(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                                batch_first=True, bidirectional=True)
        elif self.temp_unit == 'gru':
            self.gru = nn.GRU(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                              batch_first=True, bidirectional=False)
        elif self.temp_unit == 'bgru':
            self.bgru = nn.GRU(input_size=self.num_channels, hidden_size=tc_hidden, num_layers=1,
                              batch_first=True, bidirectional=True)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # shape of features: (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(1, self.timestep + 1):
            idx = (t_samples + i).long()
            encode_samples[i - 1] = z_aug2[:, idx, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        if self.temp_unit == 'tsfm':
            c_t = self.seq_transformer(forward_seq)
        elif self.temp_unit == 'lstm':
            _, (c_t, _) = self.lstm(forward_seq)
            c_t = torch.squeeze(c_t)
        elif self.temp_unit == 'blstm':
            _, (c_t, _) = self.blstm(forward_seq)
            c_t = c_t[0, :, :]
            c_t = torch.squeeze(c_t)
        elif self.temp_unit == 'gru':
            _, c_t = self.gru(forward_seq)
            c_t = torch.squeeze(c_t)
        elif self.temp_unit == 'bgru':
            _, c_t = self.bgru(forward_seq)
            c_t = c_t[0, :, :]
            c_t = torch.squeeze(c_t)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, c_t