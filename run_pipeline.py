# run_pipeline.py
import os
import yaml
import torch
from preprocessing.nhanes_loader import NHANESDataset
from train.train_detection import train_detection
from train.train_prediction import train_prediction

def main():
    # 1) YAML 설정 불러오기 (UTF-8 인코딩)
    cfg_path = "config.yaml"
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2) NHANES 데이터셋 로드
    print("===== NHANES 데이터셋 준비 시작 =====")
    dataset = NHANESDataset(
        demo_path=cfg['data']['demo_path'],
        smq_path=cfg['data']['smq_path'],
        spx_g_path=cfg['data']['spx_g_path'],
        spxraw_g_path=cfg['data']['spxraw_g_path'],
        smoothing_sigma=cfg['data']['smoothing_sigma'],
        patch_length=cfg['data']['patch_length'],
    )
    print("▶️ 총 샘플 수 (summary_df):", len(dataset))
    print("===== NHANES 데이터셋 준비 완료 =====\n")

    # 3) COPD Detection 학습
    print("===== COPD Detection 학습 시작 =====")
    detection_models = train_detection(dataset, cfg)
    print("===== COPD Detection 학습 완료 =====\n")

    # 4) COPD Early Prediction 학습 (검출 모델 사용 가능)
    print("===== COPD Early Prediction 학습 시작 =====")
    predictor_model = train_prediction(dataset, cfg, detection_models)
    print("===== COPD Early Prediction 학습 완료 =====")

if __name__ == "__main__":
    main()
