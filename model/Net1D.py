################################################################################
# Net1D.py
# 1D-CNN 기반 Net1D 모듈 (각 패치 단위 flow 시퀀스에서 특징 추출)
################################################################################

import torch
import torch.nn as nn

class Net1D(nn.Module):
    """
    1D-CNN으로 패치(patch_length 폭의 flow 시퀀스)에서 특징 벡터 추출
    구조:
      Conv1d -> ReLU -> Conv1d -> ReLU -> AdaptiveAvgPool1d(1)
    """
    def __init__(self, in_channels=1, channels=[16, 32], patch_length=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.out_dim = channels[1]

    def forward(self, x):
        """
        x shape: (batch_size, n_patches, 1, patch_length)
        -> (batch_size*n_patches, 1, patch_length)로 변형 후 conv 적용
        -> (batch_size, n_patches, out_dim) 반환
        """
        b, n_patches, _, patch_len = x.size()
        x = x.view(-1, 1, patch_len)           # (b*n_patches, 1, patch_length)
        feat = self.conv(x)                    # (b*n_patches, out_dim, 1)
        feat = feat.view(b, n_patches, self.out_dim)  # (b, n_patches, out_dim)
        return feat
