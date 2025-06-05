import pandas as pd
import pyreadstat
import numpy as np
import torch
from torch.utils.data import Dataset

class NHANESDataset(Dataset):
    def __init__(self,
                 demo_path:   str,
                 smq_path:    str,
                 spx_g_path:  str,
                 spxraw_g_path: str,
                 patch_length:  int,
                 smoothing_sigma: float):
        super().__init__()

        # 1) DEMO_G.xpt 읽기 → 연령 및 성별 가져오기
        demo_df, _ = pyreadstat.read_xport(demo_path, encoding='utf-8')
        demo_df = demo_df[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
        demo_df['SEX_BINARY'] = demo_df['RIAGENDR'].map({1: 1, 2: 0})
        print(">>> [DEBUG] demo_df shape:", demo_df.shape)
        print(">>> [DEBUG] demo_df head:\n", demo_df.head(3))

        # 2) SMQ_G.xpt 읽기 → 흡연 상태 (SMQ020)
        smq_df, _ = pyreadstat.read_xport(smq_path, encoding='utf-8')
        smq_df = smq_df[['SEQN', 'SMQ020']].copy()
        smq_df['SMOKING_BINARY'] = smq_df['SMQ020'].map({1: 1, 2: 0})
        smq_df['SMOKING_BINARY'] = smq_df['SMOKING_BINARY'].fillna(0).astype(int)
        print(">>> [DEBUG] smq_df shape:", smq_df.shape)
        print(">>> [DEBUG] smq_df head:\n", smq_df.head(3))

        # 3) SPX_G.xpt 읽기 → Pre‐BD FEV1, Pre‐BD FVC
        spx_g_df, _ = pyreadstat.read_xport(spx_g_path, encoding='utf-8')
        spx_g_df = spx_g_df[['SEQN', 'SPXNFEV1', 'SPXNQFVC']].copy()

        spx_g_df['SPXNFEV1']  = pd.to_numeric(spx_g_df['SPXNFEV1'],  errors='coerce')
        spx_g_df['SPXNQFVC'] = pd.to_numeric(spx_g_df['SPXNQFVC'], errors='coerce')
        spx_g_df = spx_g_df.dropna(subset=['SPXNFEV1', 'SPXNQFVC'])
        spx_g_df['FEV1_FVC'] = (spx_g_df['SPXNFEV1'] / spx_g_df['SPXNQFVC'])
        print(">>> [DEBUG] spx_g_df shape:", spx_g_df.shape)
        print(">>> [DEBUG] spx_g_df head:\n", spx_g_df.head(3))

        # 4) DEMO + SMQ 병합
        demo_smq_df = demo_df.merge(smq_df[['SEQN','SMOKING_BINARY']], on='SEQN', how='inner')
        print(">>> [DEBUG] demo_smq_df shape:", demo_smq_df.shape)
        print(">>> [DEBUG] demo_smq_df head:\n", demo_smq_df.head(3))

        # 5) DEMO_SMQ + SPX 병합 → 최종 merged_df
        merged_df = demo_smq_df.merge(spx_g_df[['SEQN','FEV1_FVC']], on='SEQN', how='inner')
        print(">>> [DEBUG] merged_df shape:", merged_df.shape)
        print(">>> [DEBUG] merged_df head:\n", merged_df.head(3))

        merged_df['LABEL_COPD'] = (merged_df['FEV1_FVC'] < 0.7).astype(int)
        self.summary_df = merged_df.reset_index(drop=True)
        print(f">>> NHANES DEMO+SMQ+SPX 요약 샘플 수: {len(self.summary_df)}")

        # --- 이하 원래 코드 유지 ---
        self.raw_path        = spxraw_g_path
        self.patch_length    = patch_length
        self.smoothing_sigma = smoothing_sigma
        self.seqn_list = self.summary_df['SEQN'].values

    # __len__, __getitem__ 등 기존 로직 그대로...
