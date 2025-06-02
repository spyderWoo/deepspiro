# debug_nhanes_merge.py

import pandas as pd
import pyreadstat

def main():
    # 1) DEMO_G.XPT 읽기
    demo_df, _ = pyreadstat.read_xport("data/DEMO_G.XPT")
    print("DEMO_G 전체 샘플 수:", len(demo_df))
    print("DEMO_G SEQN 예시 10개:", demo_df["SEQN"].dropna().unique()[:10].tolist())
    print()

    # 2) SMQ_G.XPT 읽기
    smq_df, _ = pyreadstat.read_xport("data/SMQ_G.XPT")
    print("SMQ_G 전체 샘플 수:", len(smq_df))
    print("SMQ_G SEQN 예시 10개:", smq_df["SEQN"].dropna().unique()[:10].tolist())
    print()

    # 3) SPX_G.XPT 읽기
    spx_g_df, _ = pyreadstat.read_xport("data/SPX_G.xpt", encoding="utf-8")
    print("SPX_G 전체 샘플 수:", len(spx_g_df))
    # SPX_G 에서 실제 'SEQN'은 중복 없이 고유한 참가자단위만 뽑기
    spx_unique = spx_g_df["SEQN"].dropna().unique()
    print("SPX_G 고유 SEQN 개수:", len(spx_unique))
    print("SPX_G SEQN 예시 10개:", spx_unique[:10].tolist())
    print()

    # 4) DEMO+SMQ 조인 시 얼마나 남는지
    # DEMO와 SMQ를 inner 합친 뒤 SEQN 고유 개수
    demo_smq = demo_df[['SEQN']].merge(
        smq_df[['SEQN']], on='SEQN', how='inner'
    )
    demo_smq_unique = demo_smq["SEQN"].dropna().unique()
    print("DEMO ∩ SMQ (공통 SEQN) 개수:", len(demo_smq_unique))
    print("DEMO ∩ SMQ SEQN 예시 10개:", demo_smq_unique[:10].tolist())
    print()

    # 5) DEMO+SMQ+SPX 조인 시 남는지
    demo_smq_spx = pd.DataFrame({'SEQN': demo_smq_unique}).merge(
        pd.DataFrame({'SEQN': spx_unique}), on='SEQN', how='inner'
    )
    merged_unique = demo_smq_spx["SEQN"].dropna().unique()
    print("DEMO ∩ SMQ ∩ SPX (공통 SEQN) 개수:", len(merged_unique))
    print("DEMO ∩ SMQ ∩ SPX SEQN 예시 10개:", merged_unique[:10].tolist())
    print()

if __name__ == "__main__":
    main()
