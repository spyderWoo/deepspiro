################################################################################
# run_predict.py
# 학습된 모델을 이용하여 새로운 샘플(CSV) 예측 스크립트
################################################################################

import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from utils.predict_utils import load_models, predict_copd

def main():
    parser = argparse.ArgumentParser(description="학습된 모델로 새로운 샘플 예측")
    parser.add_argument("--input_csv",  type=str, required=True, help="예측할 샘플 CSV 경로")
    parser.add_argument("--output_csv", type=str, required=True, help="예측 결과 저장 CSV 경로")
    args = parser.parse_args()

    # config 로드
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    # 모델 로드
    encoder, explainer, catboost_detection, catboost_prediction = load_models(
        checkpoint_dir=cfg['output']['checkpoint_dir'], cfg=cfg
    )

    # 샘플 로드 (CSV: 반드시 아래 컬럼 포함)
    # 예시: SEQN, age, sex, smoking, fev1fvc, flow_np_path, flow_time_np_path
    df = pd.read_csv(args.input_csv)
    results = []

    for idx, row in df.iterrows():
        # flow_np_path, flow_time_np_path 를 통해 flow 배열과 flow_time 불러오기
        flow_array = np.load(row['flow_np_path'])         # (N,) 1차원 flow 배열
        flow_time  = np.load(row['flow_time_np_path'])    # (N,) flow_time 배열
        patch_length = cfg['data']['patch_length']
        seq_len = len(flow_array)
        n_patches = int(np.ceil(seq_len / patch_length))
        padded_len = n_patches * patch_length
        pad_amount = padded_len - seq_len
        flow_padded = np.concatenate([flow_array, np.zeros(pad_amount, dtype=np.float32)])
        flow_patches = flow_padded.reshape(n_patches, patch_length)

        # Tensor 형태로 변환: (1, n_patches, 1, patch_length)
        flow_patches_tensor = torch.from_numpy(flow_patches).unsqueeze(0).unsqueeze(1).float()

        age = row['age']
        sex = row['sex']
        smoking = row['smoking']
        fev1fvc = row['fev1fvc']
        clinical_feats = np.array([[age, sex, smoking, fev1fvc]], dtype=np.float32)  # (1,4)

        # 예측
        prob_copd, prob_future = predict_copd(
            sample_flow_patches=flow_patches_tensor,
            sample_clinical_feats=clinical_feats,
            encoder=encoder,
            explainer=explainer,
            catboost_detection=catboost_detection,
            catboost_prediction=catboost_prediction,
            cfg=cfg
        )

        results.append({
            'SEQN': row['SEQN'],
            'prob_copd': prob_copd,
            'prob_future': prob_future
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print("예측 결과 저장 완료:", args.output_csv)

if __name__ == "__main__":
    main()
