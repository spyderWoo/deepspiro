# preprocessing/flow_converter.py

import numpy as np

def time_to_flow(volume: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Time–Volume curve를 받아서, 유한 차분(finite difference)으로 유량(Flow) 시계열을 계산합니다.
    Q(t) ≈ (V(t+Δt) - V(t)) / Δt
    Args:
      volume: numpy array, V(t) 시계열
      dt: float, 시간 간격(초)
    Returns:
      flow: numpy array, Flow(t) 시계열 (len = len(volume)-1)
    """
    # volume 크기 N → flow 크기 N-1
    return (volume[1:] - volume[:-1]) / dt

def construct_flow_volume(vol: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Flow–Volume 곡선 (x축: 볼륨, y축: 유량) 시계열을 만듭니다.
    Args:
      vol:  numpy array, Volume 시계열 (length N)
      flow: numpy array, Flow 시계열 (length N)
    Returns:
      fv: numpy array shape (N, 2), 각 행: [volume_i, flow_i]
    """
    return np.stack([vol, flow], axis=1)  # (N, 2)
