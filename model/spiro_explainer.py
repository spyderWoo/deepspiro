################################################################################
# spiro_explainer.py
# SpiroEncoder 출력에 Temporal Attention을 적용하고,
# 구조화 임상정보를 결합하여 CatBoost 입력 생성
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    1) 입력 시퀀스 (batch, n_patches, input_dim) → attention weight (batch, n_patches, 1)
    2) Linear → Swish → Bilinear → Linear → Softmax(패딩 무시)
    """
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, attention_dim)
        self.bilinear = nn.Bilinear(attention_dim, attention_dim, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x, seq_len):
        """
        x: (batch, n_patches, input_dim)
        seq_len: (batch,) 실제 패치 개수
        반환: att_weights (batch, n_patches, 1)
        """
        b, n_patches, d = x.size()

        # 1) Linear + Swish
        x_lin = self.linear1(x)  # (b, n_patches, attention_dim)
        x_swish = x_lin * torch.sigmoid(x_lin)

        # 2) Bilinear 연산 (self-attention)
        bil_out = self.bilinear(x_swish, x_swish).squeeze(-1)  # (b, n_patches)

        # 3) Linear2 → scores: (b, n_patches)
        scores = self.linear2(bil_out.unsqueeze(-1)).squeeze(-1)

        # padding mask 생성
        mask = torch.arange(n_patches, device=seq_len.device).expand(b, n_patches)
        mask = mask < seq_len.unsqueeze(1)  # True=유효, False=패딩

        scores_masked = scores.masked_fill(~mask, -1e9)
        att_weights = torch.softmax(scores_masked, dim=1).unsqueeze(-1)  # (b, n_patches, 1)
        return att_weights

class SpiroExplainer(nn.Module):
    """
    1) SpiroEncoder 출력 → attention → 가중합 → neural_prob
    2) neural_prob + 임상정보(4개) → CatBoost 학습/예측용 입력 생성
    """
    def __init__(self, encoder_output_dim, attention_dim, clinical_feat_dim):
        super().__init__()
        self.attention = TemporalAttention(encoder_output_dim, attention_dim)
        self.fc_logits = nn.Linear(encoder_output_dim, 1)
        self.clinical_feat_dim = clinical_feat_dim
        self.catboost_model = None  # CatBoost 모델은 별도 학습

    def forward(self, lstm_out, clinical_feats, seq_len):
        """
        lstm_out: (batch, n_patches, encoder_output_dim)
        clinical_feats: 사용 안 함 (CatBoost 입력은 별도 스크립트에서 처리)
        seq_len: (batch,)
        반환: (prob_neural, weighted_feature)
          prob_neural: (batch,) neural 예측 확률
          weighted_feature: (batch, encoder_output_dim) attention-weighted feature
        """
        # 1) attention
        att_w = self.attention(lstm_out, seq_len)         # (b, n_patches, 1)
        weighted = (lstm_out * att_w).sum(dim=1)           # (b, encoder_output_dim)

        # 2) FC → neural logit → sigmoid → prob
        logits = self.fc_logits(weighted)                  # (b,1)
        prob_neural = torch.sigmoid(logits).squeeze(-1)    # (b,)

        # 3) CatBoost 입력은 별도 스크립트에서 [neural_prob + clinical_feats]로 구성
        return prob_neural, weighted
