import os
import yaml
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from preprocessing.nhanes_loader import NHANESDataset
from model.spiro_predictor import compute_concavity

def train_prediction(cfg, encoder, explainer, catboost_detection_model):
    # ─────────────────────────────────────────────────────────────────────────
    # 수정: demo_path, smq_path도 넘겨줘야 함
    dataset = NHANESDataset(
        demo_path      = cfg['data']['demo_path'],       # 추가된 인자
        smq_path       = cfg['data']['smq_path'],        # 추가된 인자
        spx_g_path     = cfg['data']['spx_g_path'],
        spxraw_g_path  = cfg['data']['spxraw_g_path'],
        patch_length   = cfg['data']['patch_length'],
        smoothing_sigma= cfg['data']['smoothing_sigma']
    )
    # ─────────────────────────────────────────────────────────────────────────

    n_total = len(dataset)
    indices = np.arange(n_total)
    np.random.seed(0)
    np.random.shuffle(indices)
    split = int(0.8 * n_total)
    train_idx, val_idx = indices[:split], indices[split:]

    device = torch.device(cfg['train']['device'])
    encoder.eval()
    explainer.eval()

    X_train, y_train = [], []
    for i in train_idx:
        sample = dataset[i]
        flow_patches = sample['flow_patches'].unsqueeze(0)  # (1, n_patches, 1, patch_length)
        mask = sample['mask'].unsqueeze(0)
        seq_len = sample['seq_len'].unsqueeze(0)
        age = sample['age'].item()
        sex = sample['sex'].item()
        smoking = sample['smoking'].item()
        fev1fvc = sample['fev1fvc'].item()
        label = sample['label'].item()

        with torch.no_grad():
            lstm_out = encoder(flow_patches, mask, seq_len)
            prob_neural, _ = explainer(lstm_out, None, seq_len)
        neural_prob = prob_neural.cpu().numpy().item()

        # concavity 계산을 위해 원시 flow, flow_time을 얻어야 함
        # dataset 내부 메서드를 직접 호출 (예시) → 실제 구현 시 개선 필요
        # >> 여기는 demo/smq로 사용한 정보와 관계 없음. 예시에서는 dummy zero vector 사용
        concavity_vals = np.zeros(4, dtype=np.float32)

        # 향후 발병 라벨 예시: 이미 COPD(label=1)면 1, 아니면 0
        label_future = 1 if label == 1 else 0

        feat = np.concatenate([
            np.array([neural_prob, age, sex, smoking, fev1fvc], dtype=np.float32),
            concavity_vals
        ])
        X_train.append(feat)
        y_train.append(label_future)

    X_train = np.vstack(X_train)         # (n_train, 9)
    y_train = np.array(y_train, dtype=np.int32)

    # 2) CatBoostClassifier 학습 (Prediction)
    catboost_pred = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4, verbose=False)
    print(">>> CatBoost Prediction 모델 학습 시작...")
    catboost_pred.fit(X_train, y_train, verbose=False)
    print(">>> CatBoost 학습 완료.")

    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    pred_path = os.path.join(cfg['output']['checkpoint_dir'], "SpiroPredictor.cbm")
    catboost_pred.save_model(pred_path)
    print("CatBoost Prediction 모델 저장 위치:", pred_path)

    # 3) 검증 세트 평가
    X_val, y_val = [], []
    for i in val_idx:
        sample = dataset[i]
        flow_patches = sample['flow_patches'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)
        seq_len = sample['seq_len'].unsqueeze(0)
        age = sample['age'].item()
        sex = sample['sex'].item()
        smoking = sample['smoking'].item()
        fev1fvc = sample['fev1fvc'].item()
        label = sample['label'].item()

        with torch.no_grad():
            lstm_out = encoder(flow_patches, mask, seq_len)
            prob_neural, _ = explainer(lstm_out, None, seq_len)
        neural_prob = prob_neural.cpu().numpy().item()

        concavity_vals = np.zeros(4, dtype=np.float32)
        label_future = 1 if label == 1 else 0

        feat = np.concatenate([
            np.array([neural_prob, age, sex, smoking, fev1fvc], dtype=np.float32),
            concavity_vals
        ])
        X_val.append(feat)
        y_val.append(label_future)

    X_val = np.vstack(X_val)
    y_val = np.array(y_val, dtype=np.int32)
    val_preds = catboost_pred.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_preds)
    auprc = average_precision_score(y_val, val_preds)
    f1 = f1_score(y_val, (val_preds > 0.5).astype(int))

    print(f"[Prediction 검증] AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1-score: {f1:.4f}")

    return catboost_pred
