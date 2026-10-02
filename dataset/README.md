# 데이터 준비

원본 `.mat` 파일은 용량이 커서 GitHub 저장소에 포함하지 않습니다.
[데이터 배포 페이지](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 파일을 준비한 뒤 이 폴더에 넣습니다.

| 용도 | 파일명 |
|---|---|
| 학습·검증: Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` |
| 외부 평가: Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` |
| 추가 평가: Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` |

기본 실행에는 Batch 1과 Batch 2가 필요합니다. Batch 3는 추가 평가를 켰을 때만 사용합니다.
`2018-04-03_varcharge` 파일은 사용하지 않습니다.

다른 폴더에 저장한 경우에는 노트북의 `ESS_DATA_DIR` 환경변수 또는 실행 명령의 `--data-dir`로 위치를 지정합니다.
