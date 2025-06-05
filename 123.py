# 1) 먼저 pyreadstat을 설치해야 합니다.
#    (conda/pip 환경에 이미 pyreadstat이 없으면 아래 둘 중 하나를 실행)
#    conda install -c conda-forge pyreadstat
#    pip install pyreadstat

import pyreadstat

# SPXRAW_G.sas7bdat 파일 경로를 실제로 맞춰 주세요.
file_path = "data/SPXRAW_G.sas7bdat"

# row_limit을 지정해서, 처음 10개 행만 읽어오도록 합니다. 
# (파일이 워낙 크면 한 번에 다 읽지 않고 샘플만 보는 용도)
df_sample, meta = pyreadstat.read_sas7bdat(file_path, row_limit=10)

print("=== SPXRAW_G.sas7bdat 에서 읽은 컬럼명 목록 ===")
print(meta.column_names)   # <-- 실제 변수(컬럼) 이름들이 리스트로 출력됩니다.

print("\n=== 샘플 데이터 (최초 10개 row) ===")
print(df_sample.head())
