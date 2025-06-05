# evaluate/metrics.py

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

def compute_metrics(y_true, y_pred, y_prob):
    """
    분류 성능 지표 계산: AUROC, AUPRC, F1
    """
    # AUROC, AUPRC 계산
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    # F1-score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {'auroc': auroc, 'auprc': auprc, 'f1': f1}
