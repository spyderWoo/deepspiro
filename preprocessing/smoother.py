# preprocessing/smoother.py

import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(curve: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    1차원 Time–Volume 시계열(curve)에 Gaussian filter를 적용하여 부드럽게 만듭니다.
    Args:
      curve: numpy array, 원본 Time–Volume [V(t)] 시계열
      sigma: Gaussian 표준편차
    Returns:
      smoothed_curve: numpy array, 부드럽게 처리된 시계열
    """
    return gaussian_filter1d(curve, sigma=sigma)
