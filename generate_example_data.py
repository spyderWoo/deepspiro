################################################################################
# generate_example_data.py
# 테스트용 더미 CSV/NumPy 파일 생성 예시 (run_predict.py 테스트용)
################################################################################

import numpy as np
import pandas as pd
import os

def main():
    os.makedirs("example_data", exist_ok=True)
    n_samples = 5
    patch_length = 128

    records = []
    for i in range(n_samples):
        # 랜덤 flow 시퀀스 생성 (양수만)
        seq_len = np.random.randint(patch_length*2, patch_length*5)
        flow = np.abs(np.random.randn(seq_len).astype(np.float32))
        flow_time = np.arange(seq_len, dtype=np.float32)

        seqn = 1000 + i
        flow_np_path = f"example_data/flow_{seqn}.npy"
        flow_time_np_path = f"example_data/flow_time_{seqn}.npy"

        np.save(flow_np_path, flow)
        np.save(flow_time_np_path, flow_time)

        age = np.random.randint(50, 80)
        sex = np.random.choice([0, 1])
        smoking = np.random.choice([0, 1])
        fev1fvc = np.random.uniform(0.5, 0.9)

        records.append({
            'SEQN': seqn,
            'age': age,
            'sex': sex,
            'smoking': smoking,
            'fev1fvc': fev1fvc,
            'flow_np_path': flow_np_path,
            'flow_time_np_path': flow_time_np_path
        })

    df = pd.DataFrame(records)
    df.to_csv("example_data/new_samples.csv", index=False)
    print("더미 데이터 생성 완료: example_data/new_samples.csv")

if __name__ == "__main__":
    main()
