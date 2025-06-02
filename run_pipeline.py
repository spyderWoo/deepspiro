################################################################################
# run_pipeline.py
# 1) config.yaml 파싱
# 2) train_detection() 호출 → encoder, explainer, catboost_detection 반환
# 3) train_prediction() 호출 → catboost_prediction 반환
# 4) (예시) 평가 및 시각화
################################################################################

import os
import yaml
import numpy as np
import torch

from train.train_detection import train_detection
from train.train_prediction import train_prediction
from evaluate.metrics import compute_metrics
from evaluate.visualize import plot_roc_pr, plot_shap_summary

def main():
    # 1) config 로드 (UTF-8 인코딩 지정)
    with open("config.yaml", 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)


    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['output']['figures_dir'], exist_ok=True)

    # 2) COPD Detection 학습
    print("===== COPD Detection 학습 시작 =====")
    encoder, explainer, catboost_detection = train_detection(cfg)
    print("===== COPD Detection 학습 완료 =====\n")

    # 3) COPD Early-Prediction 학습
    print("===== COPD Early-Prediction 학습 시작 =====")
    catboost_prediction = train_prediction(cfg, encoder, explainer, catboost_detection)
    print("===== COPD Early-Prediction 학습 완료 =====\n")

    # 4) (선택) 추가 평가 및 시각화 예시
    #    train_detection()에서 반환된 val_labels, val_preds를 파일로 저장하거나,
    #    다시 DataLoader를 순회하며 예측값을 얻어와야 함. 여기서는 예시 코드로만 남겨둡니다.

    # 예시: 추후 val_labels, val_preds가 준비되었다고 가정
    # val_labels = np.array([...])
    # val_preds  = np.array([...])
    # plot_roc_pr(val_labels, val_preds, cfg['output']['figures_dir'], prefix="detection_")

    # 예시: SHAP summary
    # feature_names = ['neural_prob', 'age', 'sex', 'smoking', 'fev1fvc']
    # shap_train_feats = np.array([...])
    # plot_shap_summary(catboost_detection, shap_train_feats, feature_names, cfg['output']['figures_dir'], prefix="detection_")

    print("=== 전체 파이프라인 완료 ===")

if __name__ == "__main__":
    main()
