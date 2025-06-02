################################################################################
# nhanes_loader.py
# NHANES 2011-2012 DEMO_G.xpt, SMQ_G.xpt, SPX_G.xpt, SPXRAW_G.sas7bdat을
# 읽어와 PyTorch Dataset으로 가공합니다.
#
# 1. DEMO_G.xpt  → 연령(RIDAGEYR), 성별(RIAGENDR)
# 2. SMQ_G.xpt   → 흡연 상태(SMQ040)
# 3. SPX_G.xpt   → Pre‐BD FEV1 (SPXNFEV1), Pre‐BD FVC (SPXNQFVC) → FEV1/FVC 비율 계산
# 4. SPXRAW_G.sas7bdat → 각 참가자의 Time–Volume 시계열(원시 곡선)
# 5. Gaussian smoothing → Time→Flow 변환 → patch 단위 분할/패딩
################################################################################

import pandas as pd
import pyreadstat
import numpy as np
import torch
from torch.utils.data import Dataset

class NHANESDataset(Dataset):
    """
    NHANES SPX 데이터를 기반으로 PyTorch Dataset을 구현합니다.
    각 아이템은 아래 키들을 포함하는 dict 형태로 반환됩니다.
      - flow_patches: (n_patches, 1, patch_length) Tensor
      - mask:         (n_patches,) Tensor
      - seq_len:      스칼라 Tensor (= n_patches)
      - age:          (1,) Tensor
      - sex:          (1,) Tensor (0=여성, 1=남성)
      - smoking:      (1,) Tensor (0=비흡연, 1=흡연)
      - fev1fvc:      (1,) Tensor (FEV1/FVC 비율)
      - label:        (1,) Tensor (0=non-COPD, 1=COPD)
    """

    def __init__(self,
                 demo_path:   str,
                 smq_path:    str,
                 spx_g_path:  str,
                 spxraw_g_path: str,
                 patch_length:  int,
                 smoothing_sigma: float):
        super().__init__()

        # ─────────────────────────────────────────────────────────────────────────
        # 1) DEMO_G.xpt 읽기 → 연령 및 성별 가져오기
        demo_df, _ = pyreadstat.read_xport(demo_path)
        demo_df = demo_df[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
        # RIAGENDR: 1=남, 2=여 → 1=남성, 0=여성으로 매핑
        demo_df['SEX_BINARY'] = demo_df['RIAGENDR'].map({1: 1, 2: 0})

        # 2) SMQ_G.xpt 읽기 → 흡연 상태 (SMQ040: 과거 5일간 흡연 여부)
        smq_df, _ = pyreadstat.read_xport(smq_path)
        smq_df = smq_df[['SEQN', 'SMQ040']].copy()
        # SMQ040: 1=예, 2=아니오, 7=모름/거절 → 1→1, 2→0, 기타 NaN 처리
        smq_df['SMOKING_BINARY'] = smq_df['SMQ040'].map({1: 1, 2: 0})
        smq_df['SMOKING_BINARY'] = smq_df['SMOKING_BINARY'].fillna(0).astype(int)

        # 3) SPX_G.xpt 읽기 → Pre‐BD FEV1 (SPXNFEV1), Pre‐BD FVC (SPXNQFVC)
        spx_g_df, _ = pyreadstat.read_xport(spx_g_path, encoding='utf-8')
        spx_g_df = spx_g_df[['SEQN', 'SPXNFEV1', 'SPXNQFVC']].copy()

        # → 문자열이 섞여 있을 수 있으므로, numeric으로 강제 변환
        spx_g_df['SPXNFEV1']  = pd.to_numeric(spx_g_df['SPXNFEV1'],  errors='coerce')
        spx_g_df['SPXNQFVC'] = pd.to_numeric(spx_g_df['SPXNQFVC'], errors='coerce')

        # NaN이 있는 행(변환 실패한 문자열) 제거
        spx_g_df = spx_g_df.dropna(subset=['SPXNFEV1', 'SPXNQFVC'])

        # FEV1/FVC 비율 계산 (이제 둘 다 float 타입)
        spx_g_df['FEV1_FVC'] = (
            spx_g_df['SPXNFEV1'] / spx_g_df['SPXNQFVC']
        )

        # 4) DEMO + SMQ + SPX 요약값 merge (inner join)
        merged_df = demo_df.merge(
            smq_df[['SEQN','SMOKING_BINARY']], on='SEQN', how='inner'
        )
        merged_df = merged_df.merge(
            spx_g_df[['SEQN','FEV1_FVC']], on='SEQN', how='inner'
        )

        # 5) COPD label: FEV1/FVC < 0.70 → label = 1, else 0
        merged_df['LABEL_COPD'] = (merged_df['FEV1_FVC'] < 0.7).astype(int)

        # DataFrame 저장
        self.summary_df = merged_df.reset_index(drop=True)
        print(f">>> NHANES DEMO+SMQ+SPX 요약 샘플 수: {len(self.summary_df)}")

        # ─────────────────────────────────────────────────────────────────────────
        # 원시 spirogram 곡선 파일 경로 & 파라미터
        self.raw_path        = spxraw_g_path
        self.patch_length    = patch_length
        self.smoothing_sigma = smoothing_sigma

        # SEQN 리스트 (순서 유지)
        self.seqn_list = self.summary_df['SEQN'].values

    def __len__(self):
        return len(self.summary_df)

    def __getitem__(self, idx):
        # 1) summary_df에서 인구통계/임상정보 가져오기
        row = self.summary_df.iloc[idx]
        seqn    = int(row['SEQN'])
        age     = float(row['RIDAGEYR'])
        sex     = int(row['SEX_BINARY'])       # 0=여성, 1=남성
        smoking = int(row['SMOKING_BINARY'])
        fev1fvc = float(row['FEV1_FVC'])
        label   = int(row['LABEL_COPD'])

        # 2) SPXRAW_G.sas7bdat에서 해당 SEQN의 Time–Volume 곡선 읽기
        curve_df    = self._load_curve_for_seqn(seqn)
        time        = curve_df['TIME'].values.astype(np.float32)         # 단위: ms
        volume_ml   = curve_df['VOLUME'].values.astype(np.float32)      # 단위: mL
        volume      = volume_ml / 1000.0                                  # mL → L

        if len(volume) == 0:
            # 비어 있는 경우 → 패딩 최소 크기
            flow      = np.zeros(1, dtype=np.float32)
            flow_time = np.array([0.0], dtype=np.float32)
        else:
            # 3) Gaussian smoothing 적용
            from preprocessing.smoother import smooth_volume
            volume_smoothed = smooth_volume(volume, sigma=self.smoothing_sigma)

            # 4) Time → Flow 변환
            from preprocessing.flow_converter import volume_to_flow
            flow, flow_time = volume_to_flow(time, volume_smoothed)

        # 5) patch 단위로 자르기 (부족한 부분 0 패딩)
        seq_len   = len(flow)
        n_patches = int(np.ceil(seq_len / self.patch_length))
        padded_len= n_patches * self.patch_length
        pad_amount= padded_len - seq_len

        flow_padded  = np.concatenate([flow, np.zeros(pad_amount, dtype=np.float32)])
        flow_patches = flow_padded.reshape(n_patches, self.patch_length)

        # 6) mask 생성 (1=유효, 0=패딩) → 여기서는 모두 1로 둠
        mask = np.ones(n_patches, dtype=np.float32)

        # 7) PyTorch Tensor 변환
        flow_patches_tensor = torch.from_numpy(flow_patches).unsqueeze(1)  # (n_patches, 1, patch_length)
        mask_tensor         = torch.from_numpy(mask)                       # (n_patches,)
        seq_len_tensor      = torch.tensor(n_patches, dtype=torch.long)    # 스칼라

        age_tensor    = torch.tensor(age, dtype=torch.float32).unsqueeze(0)     # (1,)
        sex_tensor    = torch.tensor(sex, dtype=torch.int64).unsqueeze(0)       # (1,)
        smoking_tensor= torch.tensor(smoking, dtype=torch.int64).unsqueeze(0)   # (1,)
        fev1fvc_tensor= torch.tensor(fev1fvc, dtype=torch.float32).unsqueeze(0) # (1,)
        label_tensor  = torch.tensor(label, dtype=torch.float32).unsqueeze(0)   # (1,)

        return {
            'flow_patches': flow_patches_tensor,
            'mask':         mask_tensor,
            'seq_len':      seq_len_tensor,
            'age':          age_tensor,
            'sex':          sex_tensor,
            'smoking':      smoking_tensor,
            'fev1fvc':      fev1fvc_tensor,
            'label':        label_tensor
        }

    def _load_curve_for_seqn(self, seqn):
        """
        SPXRAW_G.sas7bdat에서 해당 SEQN만 필터하여 DataFrame(TIME, VOLUME)으로 반환
        chunk 단위로 읽어 메모리 절약
        """
        reader = pyreadstat.read_file_in_chunks(
            pyreadstat.read_sas7bdat, self.raw_path, chunksize=50000
        )
        for chunk_df, _ in reader:
            filtered = chunk_df[chunk_df['SEQN'] == seqn]
            if not filtered.empty:
                return filtered[['TIME', 'VOLUME']].sort_values(
                    by='TIME'
                ).reset_index(drop=True)

        # 해당 SEQN이 없으면 빈 DataFrame 반환
        return pd.DataFrame({'TIME': [], 'VOLUME': []})


if __name__ == "__main__":
    # 단독 실행 시 요약 정보 및 예시 샘플 출력
    import yaml
    with open("config.yaml", 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = NHANESDataset(
        demo_path      = cfg['data']['demo_path'],
        smq_path       = cfg['data']['smq_path'],
        spx_g_path     = cfg['data']['spx_g_path'],
        spxraw_g_path  = cfg['data']['spxraw_g_path'],
        patch_length   = cfg['data']['patch_length'],
        smoothing_sigma= cfg['data']['smoothing_sigma']
    )
    print("총 샘플 수:", len(ds))
    sample = ds[0]
    print("샘플 키:", sample.keys())
    print("flow_patches shape:", sample['flow_patches'].shape)
    print("mask shape:", sample['mask'].shape)
    print("seq_len:", sample['seq_len'])
    print("age, sex, smoking, fev1fvc, label:",
          sample['age'], sample['sex'], sample['smoking'], sample['fev1fvc'], sample['label'])
