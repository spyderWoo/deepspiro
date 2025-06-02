################################################################################
# spiro_encoder.py
# Net1D + Bi-LSTM 기반 SpiroEncoder 구현
################################################################################

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.Net1D import Net1D

class SpiroEncoder(nn.Module):
    """
    1) Net1D으로 패치별 특징 추출
    2) Bi-LSTM으로 시퀀스 전체 시계열 특징 학습
    """
    def __init__(self, net1d_channels=[16, 32], lstm_hidden_size=32, patch_length=128):
        super().__init__()
        self.net1d = Net1D(in_channels=1, channels=net1d_channels, patch_length=patch_length)
        self.lstm = nn.LSTM(
            input_size=self.net1d.out_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.out_dim = 2 * lstm_hidden_size

    def forward(self, flow_patches, mask, seq_len):
        """
        flow_patches: (batch, n_patches, 1, patch_length)
        mask: (batch, n_patches)
        seq_len: (batch,) 실제 패치 개수
        반환: lstm_out (batch, n_patches, 2*lstm_hidden_size)
        """
        # 1) Net1D
        patch_feats = self.net1d(flow_patches)  # (b, n_patches, out_dim)

        # 2) pack_padded_sequence → Bi-LSTM → pad_packed_sequence
        packed = pack_padded_sequence(
            patch_feats,
            lengths=seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return lstm_out  # (batch, n_patches, 2*lstm_hidden_size)
