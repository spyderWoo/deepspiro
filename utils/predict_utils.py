################################################################################
# predict_utils.py
# 학습된 모델 로드 및 새로운 샘플(CSV) 예측 유틸리티 함수
################################################################################

import os
import torch
import numpy as np
import pandas as pd
from model.spiro_encoder import SpiroEncoder
from model.spiro_explainer import SpiroExplainer
from model.spiro_predictor import compute_concavity_features
from catboost import CatBoostClassifier

def load_models(checkpoint_dir, cfg):
    """
    checkpoint_dir: 'weights' 폴더 경로
    cfg: config dict
    반환: encoder, explainer, catboost_detection, catboost_prediction
    """
    device = torch.device(cfg['train']['device'])

    # 1) SpiroEncoder 로드
    encoder = SpiroEncoder(
        net1d_channels=cfg['model']['net1d_channels'],
        lstm_hidden=cfg['model']['lstm_hidden_size']
    ).to(device)
    encoder_path = os.path.join(checkpoint_dir, "SpiroEncoder.pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    # 2) SpiroExplainer 인스턴스 생성 (네트워크 부분만 사용)
    explainer = SpiroExplainer(
        encoder_dim=encoder.encoder_dim,
        demo_dim=3,
        hidden_dim=cfg['model']['explainer_hidden_dim']
    ).to(device)
    explainer.eval()

    # 3) CatBoost Detection 모델 로드
    det_path = os.path.join(checkpoint_dir, "SpiroExplainer.cbm")
    catboost_detection = CatBoostClassifier()
    catboost_detection.load_model(det_path)

    # 4) CatBoost Prediction 모델 로드
    pred_path = os.path.join(checkpoint_dir, "SpiroPredictor.cbm")
    catboost_prediction = CatBoostClassifier()
    catboost_prediction.load_model(pred_path)

    return encoder, explainer, catboost_detection, catboost_prediction

def predict_copd(sample_flow_patches, sample_clinical_feats,
                 encoder, explainer, catboost_detection, catboost_prediction, cfg):
    """
    sample_flow_patches: torch.Tensor (1, n_patches, 1, patch_length)
    sample_clinical_feats: numpy array (1, 4) [age, sex, smoking, fev1fvc]
    encoder, explainer, catboost_detection, catboost_prediction: 로드된 모델
    cfg: config dict
    반환: (prob_copd, prob_future)
    """
    device = torch.device(cfg['train']['device'])
    encoder.eval()
    explainer.eval()

    # 1) Neural 탐지 확률 계산
    with torch.no_grad():
        sample_flow_patches = sample_flow_patches.to(device)
        mask = torch.ones((1, sample_flow_patches.size(1)), dtype=torch.float32).to(device)
        enc_feat = encoder(sample_flow_patches, mask)
        age = torch.tensor(sample_clinical_feats[:,0:1], dtype=torch.float32).to(device)
        sex = torch.tensor(sample_clinical_feats[:,1:2], dtype=torch.float32).to(device)
        smoking = torch.tensor(sample_clinical_feats[:,2:3], dtype=torch.float32).to(device)
        logit = explainer(enc_feat, age, sex, smoking)
        prob_neural = torch.sigmoid(logit)

    neural_prob = prob_neural.cpu().numpy().item()

    # 2) CatBoost Detection 예측 (neural_prob + 임상 정보)
    detection_input = np.concatenate([
        np.array([[neural_prob]]),    # (1,1)
        sample_clinical_feats         # (1,4)
    ], axis=1)  # 최종 (1,5)
    prob_copd = catboost_detection.predict_proba(detection_input)[:,1].item()

    # 3) Concavity 계산 → CatBoost Prediction 예측
    concavity_vals = compute_concavity_features(sample_flow_patches, cfg['data']['patch_length']).cpu().numpy()[0]
    pred_input = np.concatenate([
        np.array([[neural_prob, *sample_clinical_feats.flatten()]]),  # (1,5)
        concavity_vals.reshape(1,4)                                   # (1,4)
    ], axis=1)  # (1,9)
    prob_future = catboost_prediction.predict_proba(pred_input)[:,1].item()

    return prob_copd, prob_future
