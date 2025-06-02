################################################################################
# smoother.py
# 1차원 Gaussian filter를 사용해 volume 곡선을 부드럽게(smoothing) 처리
################################################################################

import numpy as np
from scipy.ndimage import gaussian_filter1d

def smooth_volume(volume: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    입력:
      volume: 1D numpy array (단위: L)
      sigma: Gaussian smoothing 표준편차
    출력:
      동일한 길이의 부드럽게 처리된 volume (numpy array)
    """
    return gaussian_filter1d(volume, sigma=sigma, mode='nearest')
