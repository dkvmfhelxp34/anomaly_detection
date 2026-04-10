# codex_scr

원본 `scr` 기능을 1:1로 유지한 정리본입니다.

- 기능 축소 없이, 실행 경로 의존성(`os.chdir` 하드코딩)만 로컬 폴더 기준으로 정리했습니다.
- `pot_scaler`는 `pot_configs.json`에서 `data_name/pair` 키로 로드/저장됩니다.

## 주요 파일
- `0_make_dataframe_univariate.py`
- `1_bi-lstm_train.py`
- `1_bi-lstm_test.py`
- `pot.py`
- `df_utils.py`
- `fig_utils.py`

## POT 설정
`pot_configs.json` 예시:
```json
{
  "pot_scaler": {
    "SMB1_AIR/solar": 1.0
  }
}
```
