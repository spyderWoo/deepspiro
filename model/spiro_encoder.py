# model/spiro_encoder.py

import torch
import torch.nn as nn

class Net1D(nn.Module):
    """
    1D-CNN (Net1D) 부분: DeepSpiro 논문 참고. 
    간단히 Conv1D + ReLU + MaxPool 순으로 구성.
    """

    def __init__(self, in_channels=1, channels=[16, 32], kernel_size=5, pool_size=2):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.append(nn.Conv1d(prev_ch, ch, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size))
            prev_ch = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B * n_patches, 1, patch_length)
        return self.net(x)  # → (B*n_patches, C_out, patch_length/(2^len(channels)))

class SpiroEncoder(nn.Module):
    """
    SpiroEncoder: 
      1) 각 patch마다 Net1D로 spatial feature 추출
      2) patch sequence 길이가 variable → mask와 함께 Bi-LSTM으로 통과
      3) 최종 hidden state → encoder feature (batch × encoder_dim) 반환
    """

    def __init__(self, net1d_channels=[16, 32], lstm_hidden=32):
        super().__init__()
        self.net1d = Net1D(in_channels=1, channels=net1d_channels, kernel_size=5, pool_size=2)
        # Net1D 출력 채널:
        final_c = net1d_channels[-1]
        # (pooling 후 patch_length/(2^len(channels))) 차원 → flatten을 위해 임시 계산
        # 예: patch_length=128, len(channels)=2, pool_size=2 → output_len = 128/(2^2)=32 
        # → net1d output shape = (B*n_patches, final_c, output_len)
        # LSTM 입력 차원 = final_c * output_len
        self.patch_length = 128  # config에서 동일하게 설정됨
        self.output_len = self.patch_length // (2 ** len(net1d_channels))  # 128/(2^2)=32
        self.lstm_input_dim = final_c * self.output_len

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Bi-LSTM → 양방향 → hidden_dim * 2
        self.encoder_dim = lstm_hidden * 2

    def forward(self, flow_patches, mask):
        """
        Args:
          flow_patches: Tensor (B, n_patches, 1, patch_length)
          mask:         Tensor (B, n_patches), 1=valid patch, 0=padding
        Returns:
          enc_feat: Tensor (B, encoder_dim)
        """
        B, n_patches, _, patch_len = flow_patches.size()
        # 1) Net1D 처리: batch 차원을 합쳐서 (B*n_patches, 1, patch_len)
        x = flow_patches.view(B * n_patches, 1, patch_len)
        x = self.net1d(x)  # (B*n_patches, final_c, output_len)

        # 2) flatten each patch: (B*n_patches, final_c*output_len)
        x = x.view(B, n_patches, -1)  # (B, n_patches, lstm_input_dim)

        # 3) Bi-LSTM에 mask를 적용하려면 pack_padded_sequence를 사용해야 정확하지만,
        #    여기선 단순히 패딩된 부분의 값을 0으로 두는 식으로 처리
        #    → 간단히 "mask"를 1/0로 곱해 줌
        x = x * mask.unsqueeze(-1)  # 패딩 자리 0으로 만들기

        # 4) Bi-LSTM
        out, (h_n, c_n) = self.lstm(x)  
        # h_n: (num_layers*2, B, lstm_hidden) → 양방향 concat
        # 마지막 레이어의 양방향 hidden state들 합치기
        h_forward = h_n[-2, :, :]  # (B, lstm_hidden)
        h_backward = h_n[-1, :, :] # (B, lstm_hidden)
        h = torch.cat([h_forward, h_backward], dim=1)  # (B, lstm_hidden*2)

        return h  # Encoder feature (B, encoder_dim)
