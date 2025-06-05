# evaluate/visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

def plot_roc(y_true, y_prob, save_path="roc.png"):
    """
    ROC 곡선을 그리고 저장합니다.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.4f}')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_pr(y_true, y_prob, save_path="pr.png"):
    """
    Precision-Recall 곡선을 그리고 저장합니다.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc_value = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f'AUPRC = {auprc_value:.4f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
