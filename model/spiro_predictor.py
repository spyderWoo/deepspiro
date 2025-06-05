# model/spiro_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_concavity_features(flow_patches, patch_length):
    """
    각 patch마다 concavity 지표 계산 → 4개 구간(PEF–FEF25, FEF25–FEF50, FEF50–FEF75, FEF75+)별 directed area 계산
    Args:
      flow_patches: Tensor (B, n_patches, 1, patch_length)
      patch_length: int, patch 길이
    Returns:
      concav_features: Tensor (B, 4)
    """
    B, n_patches, _, L = flow_patches.size()
    # Flow–Volume 곡선에서 patch들은 시퀀스상 concat되어 있음 → 전체 길이 = n_patches * patch_length
    # 간단하게 “각 patch” 별 최대 concavity 정도를 4개 patch 단위로 요약하겠음
    # 예시: n_patches >= 4 라면 첫 4개 patch 각각의 concavity 지표 계산
    #     n_patches < 4인 경우엔 부족한 patch를 0으로 채움

    # 1) patch 하나당 concavity 지표: directed area
    #    concavity = Σ (baseline(v) - flow(v))  (볼륨–유량 곡선이 baseline 아래에 있을수록 양수)
    concav_per_patch = []
    for i in range(n_patches):
        # patch i의 flow 시계열: (1, patch_length)
        patch_flow = flow_patches[:, i, 0, :]  # (B, patch_length)
        # 볼륨은 “시간 시퀀스 인덱스”로 가정 → 0~(patch_length-1)
        v = torch.arange(L, dtype=torch.float32, device=patch_flow.device).unsqueeze(0).repeat(B, 1)  # (B, L)
        f = patch_flow  # (B, L)

        # baseline: 시작점(0, f(0)), 끝점(L-1, f(L-1))
        m = (f[:, -1] - f[:, 0]) / (L - 1)  # (B,)
        b = f[:, 0]  # (B,)

        # directed area: Σ [ (m * v + b) - f ] over v=0..L-1
        baseline = m.unsqueeze(1) * v + b.unsqueeze(1)  # (B, L)
        diff = baseline - f                             # (B, L)
        concav_val = torch.sum(diff, dim=1)             # (B,)  # patch별 concavity
        concav_per_patch.append(concav_val.unsqueeze(1))  # list of (B,1)

    if n_patches == 0:
        # 모든 concavity가 0
        concav_tensor = torch.zeros((B, 4), device=flow_patches.device)
        return concav_tensor

    concav_all = torch.cat(concav_per_patch, dim=1)  # (B, n_patches)

    # 2) 최종 4차원 벡터: 첫 4개 patch의 concavity (존재하지 않으면 0 padding)
    if n_patches >= 4:
        feat = concav_all[:, :4]  # (B,4)
    else:
        # 부족한 개수를 0으로 패딩
        padded = torch.zeros((B, 4), device=flow_patches.device)
        padded[:, :n_patches] = concav_all
        feat = padded  # (B,4)

    return feat


class SpiroPredictor(nn.Module):
    """
    SpiroPredictor:
      - 입력: age(1), sex(1), smoking(1), concavity(4) 총 7차원 → FC → BCEWithLogitsLoss
    """

    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, age, sex, smoking, concav):
        """
        Args:
          age, sex, smoking:   Tensors (B,1)
          concav:              Tensor  (B,4)
        Returns:
          y_logit: Tensor (B,)
        """
        x = torch.cat([age, sex, smoking, concav], dim=1)  # (B,7)
        x = F.relu(self.fc1(x))        # (B, hidden_dim)
        x = F.relu(self.fc2(x))        # (B, hidden_dim)
        y_logit = self.fc3(x)          # (B,1)
        return y_logit.squeeze(1)      # (B,)
