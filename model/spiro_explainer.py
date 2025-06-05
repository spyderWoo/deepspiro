# model/spiro_explainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpiroExplainer(nn.Module):
    """
    SpiroExplainer: 
      - Encoder feature (B, encoder_dim) + DEMO 정보 (B,3) → FC → 최종 COPD logit
      - sigmoid를 사용하여 확률 반환
    """

    def __init__(self, encoder_dim=64, demo_dim=3, hidden_dim=64):
        super().__init__()
        # encoder feature → hidden_dim
        self.fc_enc = nn.Linear(encoder_dim, hidden_dim)
        # DEMO(age, sex, smoking) → hidden_dim
        self.fc_demo = nn.Linear(demo_dim, hidden_dim)
        # 두 벡터 합친 후 → hidden_dim → 1 (로그릿)
        self.fc_combined = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, enc_feat, age, sex, smoking):
        """
        Args:
          enc_feat: Tensor (B, encoder_dim)
          age:      Tensor (B,1)
          sex:      Tensor (B,1)
          smoking:  Tensor (B,1)
        Returns:
          y_logit:  Tensor (B,1)  # BCEWithLogitsLoss를 위해 로짓 반환
        """
        B = enc_feat.size(0)
        # 1) encoder feature
        x_enc = F.relu(self.fc_enc(enc_feat))  # (B, hidden_dim)
        # 2) demo feature
        demo_cat = torch.cat([age, sex, smoking], dim=1)  # (B,3)
        x_demo = F.relu(self.fc_demo(demo_cat))          # (B, hidden_dim)

        # 3) 합치기
        x = x_enc + x_demo                            # (B, hidden_dim)
        x = F.relu(self.fc_combined(x))               # (B, hidden_dim)
        y_logit = self.fc_out(x)                      # (B,1)

        return y_logit.squeeze(1)                     # (B,)
