################################################################################
# visualize.py
# ROC/PR 곡선, Attention heatmap, SHAP summary, Concavity trend 플롯 함수
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import shap

def plot_roc_pr(y_true, y_prob, output_dir, prefix=""):
    """
    y_true: 실제 레이블 (1D array)
    y_prob: 예측 확률 (1D array)
    output_dir: 결과 저장 폴더
    prefix: 파일명 접두사
    """
    os.makedirs(output_dir, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}roc_curve.png"))
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={average_precision_score(y_true, y_prob):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}pr_curve.png"))
    plt.close()

def plot_attention_heatmap(flow, att_weights, output_dir, prefix=""):
    """
    flow: 1D array (Volume–Flow 값)
    att_weights: 1D array (Attention weight; 길이 == flow 길이)
    output_dir: 결과 저장 폴더
    prefix: 파일명 접두사
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))

    # Volume-Flow 곡선
    plt.subplot(2,1,1)
    plt.plot(flow, label="Flow")
    plt.ylabel("Flow (L/ms)")
    plt.title("Volume-Flow Curve")
    plt.legend()

    # Attention weight heatmap
    plt.subplot(2,1,2)
    plt.plot(att_weights, color='r', label="Attention Weight")
    plt.xlabel("Time Index")
    plt.ylabel("Attention")
    plt.title("Attention Heatmap")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}attention_heatmap.png"))
    plt.close()

def plot_shap_summary(catboost_model, X_train, feature_names, output_dir, prefix=""):
    """
    catboost_model: 학습된 CatBoostClassifier
    X_train: (n_samples, n_features) 학습에 사용된 특성행렬
    feature_names: 특성 이름 리스트
    output_dir: 결과 저장 폴더
    prefix: 파일명 접두사
    """
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.TreeExplainer(catboost_model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(output_dir, f"{prefix}shap_summary.png"), bbox_inches='tight')
    plt.close()

def plot_concavity_trends(concavity_matrix, labels, output_dir, prefix=""):
    """
    concavity_matrix: (n_samples, 4) 각 샘플별 4개 구간 concavity 값
    labels: (n_samples,) 각 샘플의 라벨 (0=비COPD, 1=고위험/발병)
    output_dir: 결과 저장 폴더
    prefix: 파일명 접두사
    """
    os.makedirs(output_dir, exist_ok=True)
    conc_matrix = np.array(concavity_matrix)
    labels = np.array(labels)
    conc_non = conc_matrix[labels == 0]
    conc_pos = conc_matrix[labels == 1]
    mean_non = conc_non.mean(axis=0)
    mean_pos = conc_pos.mean(axis=0)

    phases = ['PEF-FEF25', 'FEF25-FEF50', 'FEF50-FEF75', 'FEF75+']
    x = np.arange(len(phases))

    plt.figure()
    plt.plot(x, mean_non, marker='o', label='비COPD 평균')
    plt.plot(x, mean_pos, marker='s', label='COPD/고위험 평균')
    plt.xticks(x, phases)
    plt.xlabel("Spirometry Phase")
    plt.ylabel("Concavity (Directed Area)")
    plt.title("Concavity Trend 비교")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}concavity_trend.png"))
    plt.close()
