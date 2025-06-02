################################################################################
# spiro_predictor.py
# Volume–Flow 곡선을 4개 구간으로 나누고 Baseline-directed area 방식으로 concavity 계산
################################################################################

import numpy as np

def compute_concavity(flow: np.ndarray, flow_time: np.ndarray, fev1fvc: float) -> np.ndarray:
    """
    flow: 1D numpy array (volume–flow 값)
    flow_time: 1D numpy array (flow 샘플의 시간, ms)
    fev1fvc: 피험자별 FEV1/FVC 비율 (scalar)
    출력: concavity_vals (길이 4) 각 구간별 concavity 값
    """
    # 1) PEF: 최대 flow 위치
    idx_pef = np.argmax(flow)

    # 2) FEF25, FEF50, FEF75 값 추정 (퍼센타일 기반 예시)
    fef25_val = np.percentile(flow, 75)
    fef50_val = np.percentile(flow, 50)
    fef75_val = np.percentile(flow, 25)

    idx_fef25 = np.argmin(np.abs(flow - fef25_val))
    idx_fef50 = np.argmin(np.abs(flow - fef50_val))
    idx_fef75 = np.argmin(np.abs(flow - fef75_val))

    # 중복 제거 및 정렬
    idxs = np.unique([idx_pef, idx_fef25, idx_fef50, idx_fef75, len(flow)-1])
    idxs = np.sort(idxs)
    if len(idxs) < 5:
        # fallback: 0, 25%, 50%, 75%, end
        idxs = np.array([0, int(len(flow)*0.25), int(len(flow)*0.50), int(len(flow)*0.75), len(flow)-1])

    concavity_vals = []
    for i in range(4):
        start, end = int(idxs[i]), int(idxs[i+1])
        if end <= start:
            concavity_vals.append(0.0)
            continue

        t0, f0 = flow_time[start], flow[start]
        t1, f1 = flow_time[end], flow[end]
        m = (f1 - f0) / (t1 - t0 + 1e-6)
        t_seg = flow_time[start:end+1]
        f_seg = flow[start:end+1]
        bl = m * (t_seg - t0) + f0
        diff = bl - f_seg
        concavity_vals.append(np.sum(diff).astype(np.float32))

    return np.array(concavity_vals, dtype=np.float32)

if __name__ == "__main__":
    # 예시
    flow = np.array([0.0, 0.5, 1.2, 1.0, 0.8, 0.3, 0.0], dtype=np.float32)
    flow_time = np.arange(len(flow), dtype=np.float32)
    fev1fvc = 0.65
    conc = compute_concavity(flow, flow_time, fev1fvc)
    print("Concavity (4개 구간):", conc)
