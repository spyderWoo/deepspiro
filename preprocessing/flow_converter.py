################################################################################
# flow_converter.py
# smoothing된 Time–Volume 곡선을 finite difference 방식으로 미분하여 Flow 생성
################################################################################

import numpy as np

def volume_to_flow(time: np.ndarray, volume: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    입력:
      time: 1D numpy array (단위: ms)
      volume: 1D numpy array (단위: L)
    출력:
      flow: 1D numpy array (단위: L/ms)
      flow_time: 1D numpy array (단위: ms)
    """
    delta_v = volume[1:] - volume[:-1]
    delta_t = time[1:] - time[:-1]
    # delta_t이 0인 경우 방지
    delta_t = np.where(delta_t == 0, 1e-6, delta_t)

    flow = delta_v / delta_t
    flow_time = time[1:].copy()
    return flow.astype(np.float32), flow_time.astype(np.float32)
