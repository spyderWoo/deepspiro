# preprocessing/nhanes_loader.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pyreadstat

from .smoother import gaussian_smooth
from .flow_converter import time_to_flow, construct_flow_volume

class NHANESDataset(Dataset):
    """
    NHANES SPX_G & SPXRAW_G 기반으로 DeepSpiro용 데이터셋을 구성합니다.
    - DEMO (DEMO_G.xpt): SEQN, RIDAGEYR(나이), RIAGENDR(성별)
    - SMQ  (SMQ_G.xpt): SEQN, SMQ020(흡연 여부)
    - SPX_G (SPX_G.xpt): SEQN, SPXNFEV1, SPXNFVC (Pre-BD)
    - SPXRAW_G (SPXRAW_G.sas7bdat): SEQN별 raw curve('SPXRAW') ->  시간–부피 시계열
    """

    def __init__(self,
                 demo_path: str,
                 smq_path: str,
                 spx_g_path: str,
                 spxraw_g_path: str,
                 smoothing_sigma: float = 2.0,
                 patch_length: int = 128):
        super().__init__()
        self.patch_length = patch_length
        self.smoothing_sigma = smoothing_sigma

        # 1) DEMO 데이터
        df_demo = pd.read_xport(demo_path)[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
        df_demo.rename(columns={'RIDAGEYR':'AGE', 'RIAGENDR':'GENDER'}, inplace=True)
        # GENDER: 1=Male, 2=Female → 0/1 인코딩
        df_demo['SEX'] = (df_demo['GENDER'] == 1).astype(int)

        # 2) SMQ 데이터
        df_smq = pd.read_xport(smq_path)[['SEQN', 'SMQ020']]
        # SMQ020: 1=Yes, 2=No, NaN=No
        df_smq['SMOKING'] = df_smq['SMQ020'].apply(lambda x: 1 if x == 1.0 else 0).astype(int)
        df_smq = df_smq[['SEQN', 'SMOKING']]

        # 3) SPX_G 데이터 (Pre-BD 요약값)
        #    SPXNFEV1, SPXNFVC: 밀리리터 단위 → 리터 단위로 변환
        df_spx_g = pd.read_xport(spx_g_path)[['SEQN', 'SPXNFEV1', 'SPXNFVC']]
        df_spx_g.rename(columns={'SPXNFEV1':'FEV1_ml', 'SPXNFVC':'FVC_ml'}, inplace=True)
        df_spx_g['FEV1'] = df_spx_g['FEV1_ml'] / 1000.0
        df_spx_g['FVC'] = df_spx_g['FVC_ml'] / 1000.0
        df_spx_g['FEV1_FVC'] = df_spx_g['FEV1'] / df_spx_g['FVC']
        df_spx_g = df_spx_g[['SEQN', 'FEV1_FVC']]

        # 4) 세 테이블(DEMO, SMQ, SPX_G) 병합 → 레이블 생성
        #    - Self-report 정보 대신 일단 FEV1/FVC < 0.70 → COPD 확진(1), else (0)
        df_demo_smq = pd.merge(df_demo, df_smq, on='SEQN', how='inner')
        df_summary = pd.merge(df_demo_smq, df_spx_g, on='SEQN', how='inner')
        # 레이블: FEV1_FVC < 0.70 → COPD(1), else 0
        df_summary['LABEL'] = (df_summary['FEV1_FVC'] < 0.70).astype(int)

        # 5) SPXRAW_G 데이터 (raw 시계열) 로드
        #    컬럼: ['SEQN', 'SPXRAW'] 에서 SPXRAW는 “콤마 구분된 문자열” 형태
        if os.path.exists(spxraw_g_path):
            df_raw, _ = pyreadstat.read_sas7bdat(spxraw_g_path)
            df_raw = df_raw[['SEQN', 'SPXRAW']].dropna(subset=['SPXRAW'])
            # SPXRAW: 문자열(byte → str) 형태이므로, bytes→str 로 변환
            # 예: b"0,1,2,3,..." → "0,1,2,3,..." 로 변환 후 np.fromstring
            def parse_raw_string(x):
                if isinstance(x, bytes):
                    s = x.decode('utf-8')
                else:
                    s = str(x)
                # 공백 제거 후 콤마(,) 구분
                return np.fromstring(s.replace(" ", ""), sep=',', dtype=np.float32)

            df_raw['CURVE'] = df_raw['SPXRAW'].apply(parse_raw_string)
        else:
            df_raw = pd.DataFrame(columns=['SEQN', 'SPXRAW', 'CURVE'])

        # 6) df_summary와 df_raw를 SEQN 기준으로 병합 (outer join → raw 없으면 NaN)
        df_full = pd.merge(df_summary, df_raw[['SEQN', 'CURVE']], on='SEQN', how='left')
        # CURVE NaN은 빈 배열로 대체
        df_full['CURVE'] = df_full['CURVE'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([], dtype=np.float32))

        # 7) 인덱스 재설정
        df_full = df_full.reset_index(drop=True)

        # 8) 클래스 변수 저장
        self.summary_df = df_full

    def __len__(self):
        return len(self.summary_df)

    def __getitem__(self, idx):
        """
        한 샘플을 반환합니다. dict 형태로 return:
        {
         'flow_patches': Tensor(n_patches, 1, patch_length),
         'mask':         Tensor(n_patches),
         'age':          Tensor(1),
         'sex':          Tensor(1),
         'smoking':      Tensor(1),
         'label':        Tensor(1)
        }
        """
        row = self.summary_df.iloc[idx]
        seqn = row['SEQN']
        age = torch.tensor([row['AGE']], dtype=torch.float32)
        sex = torch.tensor([row['SEX']], dtype=torch.float32)
        smoking = torch.tensor([row['SMOKING']], dtype=torch.float32)
        label = torch.tensor([row['LABEL']], dtype=torch.float32)

        # 1) raw curve가 있으면 부드럽게(smoothing), 시간→유량, volume→flow curve 생성
        curve = row['CURVE']  # numpy 1d array, 시간-부피 시계열
        if curve.size > 0:
            # Gaussian smoothing
            curve_smooth = gaussian_smooth(curve, sigma=self.smoothing_sigma)

            # 시간 간격이 일정하다고 가정 (예: 10ms 간격 → 0.01 s)
            # 실제 NHANES는 10ms 단위 → 0.01초
            delta_t = 0.01  # 10ms

            # Time→Flow (유량) 계산
            flow = time_to_flow(curve_smooth, dt=delta_t)   # 길이 L-1 (C[1:]-C[:-1])/Δt

            # Volume–Flow 곡선 생성: volume은 curve_smooth[:L-1], flow는 flow
            v = curve_smooth[:-1]
            f = flow
            flow_volume = construct_flow_volume(v, f)  # (L-1, 2) 형태

            # 2) flow_volume을 patch_length 단위로 잘라서 n_patches, 1, patch_length
            n = flow_volume.shape[0]
            patch_len = self.patch_length
            n_patches = int(np.ceil(n / patch_len))
            padded = np.zeros((n_patches * patch_len, 2), dtype=np.float32)
            padded[:n, :] = flow_volume

            patches = padded.reshape(n_patches, patch_len, 2)  # (n_patches, patch_len, 2)
            # Flow–Volume은 2차원 데이터이지만, 논문 DeepSpiro는 “유량 축”만 학습하므로 1채널로 사용
            # 채널 축 넣어서 (n_patches, 1, patch_len)
            # 여기서는 “flow” 값만(두 번째 열) 사용
            flow_patches = patches[:, :, 1]   # (n_patches, patch_len)
            flow_patches = torch.from_numpy(flow_patches).unsqueeze(1)  # (n_patches, 1, patch_len)
            mask = torch.ones(n_patches, dtype=torch.float32)
        else:
            # raw curve 없음 → 최소 하나의 0 패치 반환
            flow_patches = torch.zeros((1, 1, self.patch_length), dtype=torch.float32)
            mask = torch.zeros((1,), dtype=torch.float32)

        return {
            'flow_patches': flow_patches,  # (n_patches,1,patch_length)
            'mask': mask,                  # (n_patches,)
            'age': age,                    # (1,)
            'sex': sex,                    # (1,)
            'smoking': smoking,            # (1,)
            'label': label                 # (1,)
        }
