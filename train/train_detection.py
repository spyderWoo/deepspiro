import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from catboost import CatBoostClassifier

# 여기를 수정: NHANESDataset 임포트 경로 그대로 유지하되, 시그니처 변경 유의
from preprocessing.nhanes_loader import NHANESDataset
from model.spiro_encoder import SpiroEncoder
from model.spiro_explainer import SpiroExplainer

def train_detection(cfg):
    device = torch.device(cfg['train']['device'])

    # ────────────────────────────────────────────────────────────────────────
    # 수정: demo_path, smq_path도 넘겨줘야 함
    dataset = NHANESDataset(
        demo_path      = cfg['data']['demo_path'],       # 추가된 인수
        smq_path       = cfg['data']['smq_path'],        # 추가된 인수
        spx_g_path     = cfg['data']['spx_g_path'],
        spxraw_g_path  = cfg['data']['spxraw_g_path'],
        patch_length   = cfg['data']['patch_length'],
        smoothing_sigma= cfg['data']['smoothing_sigma']
    )
    # ────────────────────────────────────────────────────────────────────────

    # 학습/검증 분할 (80% train, 20% val) → 인덱스 섞어서 분리
    n_total = len(dataset)
    indices = np.arange(n_total)
    np.random.seed(0)
    np.random.shuffle(indices)
    split = int(0.8 * n_total)
    train_idx, val_idx = indices[:split], indices[split:]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set   = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=0)

    # 2) SpiroEncoder & SpiroExplainer 생성
    encoder = SpiroEncoder(
        net1d_channels=cfg['model']['net1d_channels'],
        lstm_hidden_size=cfg['model']['lstm_hidden_size'],
        patch_length=cfg['data']['patch_length']
    ).to(device)

    explainer = SpiroExplainer(
        encoder_output_dim=encoder.out_dim,
        attention_dim=cfg['model']['attention_dim'],
        clinical_feat_dim=4
    ).to(device)

    # Neural 부분 optimizer, loss 정의
    params = list(encoder.parameters()) + list(explainer.fc_logits.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg['train']['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss()

    # 3) Neural Network 학습 루프
    for epoch in range(cfg['train']['epochs_detection']):
        encoder.train()
        explainer.train()
        total_loss = 0.0

        for batch in train_loader:
            flow_patches = batch['flow_patches'].to(device)   # (b, n_patches, 1, patch_length)
            mask         = batch['mask'].to(device)           # (b, n_patches)
            seq_len      = batch['seq_len'].to(device)        # (b,)
            label        = batch['label'].to(device)          # (b,1)

            lstm_out = encoder(flow_patches, mask, seq_len)   # (b, n_patches, 2*lstm_hidden)
            prob_neural, _ = explainer(lstm_out, None, seq_len)  # (b,)

            # BCEWithLogitsLoss를 쓰려면 sigmoid → logit, 즉 torch.logit()
            loss = criterion(torch.logit(prob_neural), label.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{cfg['train']['epochs_detection']}] Neural Loss: {avg_loss:.4f}")

    # 4) Neural 학습 완료 후, train set 전체 순회하며 neural_prob + 구조화 임상 정보 획득 → CatBoost 학습 데이터 구성
    encoder.eval()
    explainer.eval()

    train_features, train_labels = [], []
    for batch in train_loader:
        with torch.no_grad():
            flow_patches = batch['flow_patches'].to(device)
            mask         = batch['mask'].to(device)
            seq_len      = batch['seq_len'].to(device)
            age          = batch['age'].cpu().numpy().reshape(-1, 1)
            sex          = batch['sex'].cpu().numpy().reshape(-1, 1)
            smoking      = batch['smoking'].cpu().numpy().reshape(-1, 1)
            fev1fvc      = batch['fev1fvc'].cpu().numpy().reshape(-1, 1)
            label        = batch['label'].cpu().numpy().reshape(-1, 1)

            lstm_out = encoder(flow_patches, mask, seq_len)
            prob_neural, weighted_feat = explainer(lstm_out, None, seq_len)

            neural_prob_np = prob_neural.cpu().numpy().reshape(-1, 1)
            clinical_np = np.concatenate([age, sex, smoking, fev1fvc], axis=1)  # (b,4)
            features = np.concatenate([neural_prob_np, clinical_np], axis=1)    # (b,5)

            train_features.append(features)
            train_labels.append(label)

    train_features = np.vstack(train_features)   # (n_train,5)
    train_labels   = np.vstack(train_labels).ravel()  # (n_train,)

    # 5) CatBoostClassifier 학습 (Detection)
    catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=4, verbose=False)
    print(">>> CatBoost Detection 모델 학습 시작...")
    catboost_model.fit(train_features, train_labels, verbose=False)
    print(">>> CatBoost 학습 완료.")

    # 학습된 CatBoost 모델 저장
    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    det_path = os.path.join(cfg['output']['checkpoint_dir'], "SpiroExplainer.cbm")
    catboost_model.save_model(det_path)
    print("CatBoost Detection 모델 저장 위치:", det_path)

    # 6) 검증 세트 예측 및 지표 계산
    val_features, val_labels = [], []
    for batch in val_loader:
        with torch.no_grad():
            flow_patches = batch['flow_patches'].to(device)
            mask         = batch['mask'].to(device)
            seq_len      = batch['seq_len'].to(device)
            age          = batch['age'].cpu().numpy().reshape(-1, 1)
            sex          = batch['sex'].cpu().numpy().reshape(-1, 1)
            smoking      = batch['smoking'].cpu().numpy().reshape(-1, 1)
            fev1fvc      = batch['fev1fvc'].cpu().numpy().reshape(-1, 1)
            label        = batch['label'].cpu().numpy().reshape(-1, 1)

            lstm_out = encoder(flow_patches, mask, seq_len)
            prob_neural, weighted_feat = explainer(lstm_out, None, seq_len)

            neural_prob_np = prob_neural.cpu().numpy().reshape(-1, 1)
            clinical_np = np.concatenate([age, sex, smoking, fev1fvc], axis=1)
            features = np.concatenate([neural_prob_np, clinical_np], axis=1)

            val_features.append(features)
            val_labels.append(label)

    val_features = np.vstack(val_features)   # (n_val,5)
    val_labels   = np.vstack(val_labels).ravel()  # (n_val,)

    val_preds = catboost_model.predict_proba(val_features)[:, 1]
    auroc = roc_auc_score(val_labels, val_preds)
    auprc = average_precision_score(val_labels, val_preds)
    f1 = f1_score(val_labels, (val_preds > 0.5).astype(int))

    print(f"[검증] Detection AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1-score: {f1:.4f}")

    return encoder, explainer, catboost_model
