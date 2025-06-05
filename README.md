# COPD 조기 예측 파이프라인

이 프로젝트는 미국 NHANES (National Health and Nutrition Examination Survey) 2007-2008 데이터에 기반하여 만성 폐쇄성 폐질환(COPD)을 조기에 감지하는 머신러닝 모델 학습 파이프라인입니다. 인구통계 정보(DEMO_G), 흡연 설문(SMQ_G), 호흡기 건강 설문(RDQ_G) 및 폐활량 검사(SPX_G) 데이터를 병합하여, 폐활량 검사 결과(FEV1/FVC 비율)로 정의한 COPD 여부를 예측하는 분류 모델을 학습합니다. CatBoost 머신러닝 알고리즘을 사용하며, 모델 학습 완료 후 특징 중요도 분석을 위해 SHAP 기반 설명 시각화 결과를 생성합니다.

## 파일 구조
COPD-Early-Prediction/
├── README.md
├── requirements.txt
├── config.yaml
├── run_pipeline.py
├── nhanes_loader.py
└── train_detection.py


## 설치 및 실행 방법

1. **리포지토리 클론 및 의존성 설치:** 
   - 리포지토리를 클론한 후, 프로젝트 디렉터리로 이동합니다. 
   - Python 3 환경에서 `pip install -r requirements.txt`를 실행하여 필요한 패키지를 설치합니다.

2. **데이터 준비:** 
   - NHANES 2007-2008 설문/검사 데이터 파일(DEMO_G.xpt, SMQ_G.xpt, RDQ_G.xpt, SPX_G.xpt)을 `data/` 디렉터리에 넣습니다. (`config.yaml`에서 경로와 파일 이름을 변경할 수 있습니다.)

3. **파이프라인 실행:** 
   - `config.yaml` 설정을 확인/수정한 후, 프로젝트 디렉터리에서 `python run_pipeline.py` 명령을 실행합니다.
   - 스크립트 실행 후 콘솔에 데이터셋 크기 및 모델 학습 진행 상황이 출력됩니다. 학습 완료 시 `copd_model.cbm` 모델 파일이 저장되고, `shap_summary.png` 및 `shap_feature_importance.png` 등의 설명 결과 그래프 이미지가 생성됩니다.

## 1. 필수 데이터 준비

이 저장소에는 모델 학습에 필요한 대용량 원시 spirogram 데이터(`spxraw_g.sas7bdat`)를 포함하지 않습니다.  
아래 NHANES 공식 링크에서 직접 다운로드하여 **`data/`** 폴더에 넣어주세요:

- SPXRAW_G.sas7bdat (약 441 MB):  
  https://www.cdc.gov/Nchs/Nhanes/2011-2012/SPXRAW_G.sas7bdat

- SPX_G.xpt (약 몇 MB):  
  https://www.cdc.gov/Nchs/Nhanes/2011-2012/SPX_G.XPT

- DEMO_G.xpt (약 몇 MB):  
  https://www.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT

- SMQ_G.xpt (약 몇 MB):  
  https://www.cdc.gov/Nchs/Nhanes/2011-2012/SMQ_G.XPT
