################################################################################
# metrics.py
# 모델 성능 (ROC, PR, F1, Accuracy) 계산 함수
################################################################################

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    y_true: 실제 레이블 (1D array)
    y_prob: 예측 확률 (1D array)
    threshold: 이진 분류 임계값 (기본 0.5)

    반환: {
      'AUROC': ...,
      'AUPRC': ...,
      'F1-score': ...,
      'Accuracy': ...
    }
    """
    y_pred = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    acc = (y_pred == y_true).mean()
    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1-score': f1,
        'Accuracy': acc
    }

if __name__ == "__main__":
    y_true = np.array([0,1,1,0,1,0,0,1])
    y_prob = np.array([0.1,0.8,0.6,0.4,0.9,0.2,0.3,0.7])
    metrics = compute_metrics(y_true, y_prob)
    print("계산된 성능 지표:", metrics)
