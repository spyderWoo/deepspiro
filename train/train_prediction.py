# train/train_prediction.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from evaluate.metrics import compute_metrics
from evaluate.visualize import plot_roc, plot_pr

from model.spiro_predictor import SpiroPredictor, compute_concavity_features
from model.spiro_encoder import SpiroEncoder  # 필요 시
from model.spiro_explainer import SpiroExplainer  # 필요 시

from tqdm import tqdm

def collate_fn(batch):
    """
    Detection과 동일한 collate_fn을 그대로 사용합니다.
    """
    max_n_patches = max(item['flow_patches'].shape[0] for item in batch)
    patch_len = batch[0]['flow_patches'].shape[-1]
    B = len(batch)

    flow_patches_padded = torch.zeros((B, max_n_patches, 1, patch_len), dtype=torch.float32)
    mask_padded = torch.zeros((B, max_n_patches), dtype=torch.float32)
    ages = []
    sexes = []
    smokings = []
    labels = []

    for i, item in enumerate(batch):
        n_p = item['flow_patches'].shape[0]
        flow_patches_padded[i, :n_p, :, :] = item['flow_patches']
        mask_padded[i, :n_p] = item['mask']
        ages.append(item['age'])
        sexes.append(item['sex'])
        smokings.append(item['smoking'])
        labels.append(item['label'])

    ages = torch.stack(ages, dim=0)
    sexes = torch.stack(sexes, dim=0)
    smokings = torch.stack(smokings, dim=0)
    labels = torch.stack(labels, dim=0)

    return {
        'flow_patches': flow_patches_padded,  # (B, max_n_patches, 1, patch_len)
        'mask': mask_padded,                  # (B, max_n_patches)
        'age': ages,                          # (B,1)
        'sex': sexes,                         # (B,1)
        'smoking': smokings,                  # (B,1)
        'label': labels                       # (B,1)
    }

def train_prediction(dataset, cfg, detection_models=None):
    """
    DeepSpiro 방식으로 Early Prediction 모델 학습.
    - SpiroPredictor 사용(BCEWithLogitsLoss)
    - patch별 concavity 특징 4개 사용
    """
    tp_cfg = cfg['train']
    device = torch.device(tp_cfg['device'] if torch.cuda.is_available() else 'cpu')
    batch_size = int(cfg['data']['batch_size'])
    epochs     = int(cfg['train']['epochs_prediction'])
    lr         = float(cfg['train']['learning_rate'])
    test_size  = float(cfg['train']['test_size'])
    seed       = int(cfg['train']['random_seed'])

    total_len = len(dataset)
    val_len   = int(total_len * test_size)
    train_len = total_len - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 모델 초기화: 입력 차원 = DEMO(3) + concavity(4) = 7
    predictor = SpiroPredictor(input_dim=3 + 4, hidden_dim=cfg['model']['explainer_hidden_dim']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    best_auc = 0.0
    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)

    print(f"▶️ Prediction 학습 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    total_start_time = time.time()
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        predictor.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            flow_patches = batch['flow_patches'].to(device)  # (B, n_patches,1,patch_len)
            age          = batch['age'].to(device)           # (B,1)
            sex          = batch['sex'].to(device)           # (B,1)
            smoking      = batch['smoking'].to(device)       # (B,1)
            label        = batch['label'].to(device).squeeze(1)  # (B,)

            # concavity 특징 계산 → (B,4)
            concav = compute_concavity_features(flow_patches, cfg['model']['patch_length']).to(device)

            optimizer.zero_grad()
            y_logit = predictor(age, sex, smoking, concav)  # (B,)
            loss = criterion(y_logit, label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # 검증 루프
        predictor.eval()
        y_true_list = []
        y_prob_list = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Valid]"):
                flow_patches = batch['flow_patches'].to(device)
                age          = batch['age'].to(device)
                sex          = batch['sex'].to(device)
                smoking      = batch['smoking'].to(device)
                label        = batch['label'].to(device).squeeze(1)

                concav = compute_concavity_features(flow_patches, cfg['model']['patch_length']).to(device)
                y_logit = predictor(age, sex, smoking, concav)
                y_prob = torch.sigmoid(y_logit)

                y_true_list.append(label.cpu().numpy())
                y_prob_list.append(y_prob.cpu().numpy())

        y_true_all = np.concatenate(y_true_list)
        y_prob_all = np.concatenate(y_prob_list)
        y_pred_all = (y_prob_all >= 0.5).astype(int)

        metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
        avg_loss = np.mean(epoch_losses)
        epoch_duration = time.time() - epoch_start_time

        print(f"[Epoch {epoch}/{epochs}] "
              f"loss: {avg_loss:.4f} | AUROC: {metrics['auroc']:.4f} | "
              f"AUPRC: {metrics['auprc']:.4f} | F1: {metrics['f1']:.4f} | "
              f"Epoch Time: {epoch_duration:.1f}s")

        if metrics['auroc'] > best_auc:
            best_auc = metrics['auroc']
            ckpt_path = os.path.join(cfg['output']['checkpoint_dir'], "deepspiro_prediction_best.pth")
            torch.save(predictor.state_dict(), ckpt_path)

    total_duration = time.time() - total_start_time
    print(f"▶️ 전체 Prediction 학습 완료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"▶️ 전체 Prediction 학습 시간: {total_duration:.1f}s ({total_duration/60:.2f}분)")

    # ROC/PR 곡선 저장
    plot_roc(y_true_all, y_prob_all, save_path=os.path.join(cfg['output']['figures_dir'], "roc_prediction.png"))
    plot_pr(y_true_all, y_prob_all, save_path=os.path.join(cfg['output']['figures_dir'], "pr_prediction.png"))
    print("▶️ Prediction 모델 및 ROC/PR 곡선 저장 완료")

    return predictor
