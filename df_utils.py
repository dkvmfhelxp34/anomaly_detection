import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "json_folder"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

COLUMN_CONFIG = load_json(JSON_DIR / "columns.json")
FLAG_COLUMN_MAPPING = load_json(JSON_DIR / "flag_columns.json")

def load_station_periods(json_path=None):
    if json_path is None:
        json_path = JSON_DIR / "station_periods.json"
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 문자열을 datetime 형태로 변환해주는 형태로 정리
    periods = {
        k: (pd.to_datetime(v[0]), pd.to_datetime(v[1]))
        for k, v in data.items()
    }
    return periods

# 로딩
station_periods = load_station_periods()

# -----------------------------
# 1) 센서별 / 특징별 컬럼 설정
# -----------------------------
# key: (지점명, 특징)  / 특징이 없으면 None 사용

def get_columns(station, feature):
    key = f"{station}-{feature}"
    if key in COLUMN_CONFIG:
        return COLUMN_CONFIG[key]
    raise ValueError(f"컬럼 설정이 없는 조합: {key}")

# -----------------------------
# 2) 파일 검색 관련 유틸
# -----------------------------

def get_file_list(
    base_dir: str,
    station: str,
    feature: None,
    prefix: str = "DATA_",
    suffix: str = ".csv",
):
    """
    base_dir 아래에서 station / feature 에 맞는 파일 리스트 가져오기.
    예) DATA_SMB1(해측)_Air_140101_141231.csv
    - station: "SMB1"
    - feature: "Air"
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"디렉터리가 존재하지 않습니다: {base_dir}")

    files: list[Path] = []
    for fname in os.listdir(base_dir):
        if not fname.endswith(suffix):
            continue

        if station not in fname:
            continue

        if feature is not None:
            if feature not in fname:
                continue
        # feature 없으면 station만 포함되면 됨

        files.append(base_dir / fname)

    files.sort()
    return files


# -----------------------------
# 3) 단일 파일 로딩 유틸
# -----------------------------

def load_single_csv(
    file_path: str,
    station: str,
    usecols,
    encoding: str = "cp949",
) -> pd.DataFrame:
    """
    단일 csv 파일 로딩 + 날짜 자동 인식 + station별 시간 간격 맞추기.
    SMB1~SMB6: 10분 간격
    Sea_level: 1분 간격
    """
    file_path = Path(file_path)

    # 1) 인코딩 후보 리스트
    encoding_candidates = [
        "utf-8",
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "latin1",
    ]

    last_error = None
    df_sample = None

    # 2) 인코딩 자동 탐지
    for enc in encoding_candidates:
        try:
            df_sample = pd.read_csv(file_path, encoding=enc, nrows=1)
            encoding = enc
            break
        except Exception as e:
            last_error = e
            continue

    if df_sample is None:
        raise UnicodeDecodeError(
            f"파일을 읽을 수 없습니다. Tried={encoding_candidates}. LastError={last_error}"
        )

    # 3) 전체 컬럼 확인
    full_cols = list(df_sample.columns)

    # 4) 날짜 컬럼 후보
    date_candidates = ["날짜", "obs_time", "OBS_TIME", "Obs_time"]

    date_col = None
    for dc in date_candidates:
        if dc in full_cols:
            date_col = dc
            break

    if date_col is None:
        raise ValueError(f"날짜 컬럼을 찾을 수 없습니다: {file_path}")

    # 5) usecols 매핑
    processed_usecols = []
    for c in usecols:
        if c == "날짜":
            processed_usecols.append(date_col)
        else:
            if c in full_cols:
                processed_usecols.append(c)

    # 6) 실제 CSV 로딩
    df = pd.read_csv(file_path, encoding=encoding, usecols=processed_usecols)

    # 7) 날짜 컬럼 통일
    df.rename(columns={date_col: "날짜"}, inplace=True)
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 8) 정렬
    df = df.sort_values("날짜").reset_index(drop=True)

    # 9) 여기서 station별 간격 맞추기
    if station in ["SMB1", "SMB2", "SMB3", "SMB4"]:
        rule = "10T"   # 10분
    elif station in ["SMB5", "SMB6"]:
        rule = "30T"   # 30분
    else:
        rule = "1T"    # 나머지(Sea_level 등은 1분)

    df = df.set_index("날짜")
    start = df.index.min()
    end = df.index.max()
    new_index = pd.date_range(start=start, end=end, freq=rule)
    df = df.reindex(new_index)
    df = df.reset_index().rename(columns={"index": "날짜"})

    return df

def load_obs_group(
    base_dir: str,
    station: str,
    feature: None,
    encoding: str = "cp949",
    sort_by_time: bool = True,
) -> pd.DataFrame:

    base_dir = Path(base_dir)

    # 1) 컬럼 설정 가져오기
    usecols = get_columns(station, feature)

    # 2) 파일 리스트 가져오기
    file_list = get_file_list(base_dir, station, feature)

    if not file_list:
        raise FileNotFoundError(
            f"해당 조합의 파일이 없습니다. base_dir={base_dir}, station={station}, feature={feature}"
        )

    # 3) 각 파일 읽어서 concat
    dfs: list[pd.DataFrame] = []
    for f in file_list:
        df = load_single_csv(f, station, usecols=usecols, encoding=encoding)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # 4) 날짜 정렬
    if sort_by_time and "날짜" in df_all.columns:
        df_all = df_all.sort_values("날짜").reset_index(drop=True)

    return df_all

def build_group_dataset(df1, df2, var_list, suffix1="_1", suffix2="_2"):
    """
    두 지점 df에서 var_list에 해당하는 변수들만 추출해 병합.
    var_list: ["풍속(m/s)", "최대풍속(m/s)"] 같은 변수 묶음
    """
    df1_sub = df1[["날짜"] + var_list].copy()
    df2_sub = df2[["날짜"] + var_list].copy()

    df1_sub = df1_sub.rename(columns={col: col + suffix1 for col in var_list})
    df2_sub = df2_sub.rename(columns={col: col + suffix2 for col in var_list})

    merged = df1_sub.merge(df2_sub, on="날짜", how="inner")
    return merged

# -----------------------------
# 4) Flag 데이터 로딩 및 병합 유틸 (파일받아서 수정하기)
# -----------------------------
def convert_qc1_to_label(flag_val):
    """
    Flag 값을 TranAD 라벨로 변환
    111 (정상) -> 0 (정상)
    3xx (의심), 4xx (오류) -> 그대로두기
    999 (결측) -> np.nan
    """

    if pd.isna(flag_val):
        return np.nan
    val = int(flag_val)

    if val == 111:
        return 0 
    elif val == 999:
        return np.nan
    else:
        return val

    return print('변환실패 문제발생')

def convert_qc2_to_label(flag_val):
    """
    QC2 라벨을 TranAD 라벨(정상=0, 이상=1, 결측=np.nan)로 변환

    QC2 규칙:
        정상: 1 → 0
        결측: 9 → np.nan
        이상: 4 → 1
        그 외 값은 결측 처리(np.nan)
    """
    try:
        val = int(flag_val)
    except:
        return np.nan  # 변환 실패도 결측 처리

    if val == 1:      # 정상
        return 0
    elif val == 4:    # 이상
        return 1
    elif val == 9:    # 결측
        return np.nan
    else:
        return np.nan  # 알 수 없는 값도 결측

QC_CONVERTERS = {
    "qc1": convert_qc1_to_label,
    "qc2": convert_qc2_to_label}

def read_csv_with_encoding(file_path):
    encoding_candidates = [
        "utf-8",
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "latin1"
    ]
    
    last_error = None

    for enc in encoding_candidates:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            
            # 컬럼명이 깨졌는지 검사
            bad = any(('�' in col or '癤' in col or col.strip() == "") for col in df.columns)
            if bad:
                continue  # 깨졌으면 다음 인코딩으로 다시 시도
            
            return df  # 성공
        except Exception as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        f"[인코딩 오류] 파일을 어떤 인코딩으로도 읽을 수 없습니다.\n"
        f"시도한 인코딩: {encoding_candidates}\n"
        f"마지막 오류: {last_error}"
    )


def smb_load_flag_group(base_dir, station, feature, qc_type="qc2"):
    """
    SMB1~SMB6, Air/Surface/Water 전용 Flag 로더
    """
    base_dir = Path(base_dir)
    qc_func = QC_CONVERTERS.get(qc_type)
    if qc_func is None:
        raise ValueError(f"알 수 없는 QC 타입: {qc_type}")
    
    # 1) Flag 파일 찾기
    files = []
    for fname in os.listdir(base_dir):
        if not fname.endswith(".csv"):
            continue
        if not fname.startswith(station):
            continue
        if feature and feature.lower() not in fname.lower():
            continue

        files.append(base_dir / fname)

    if not files:
        print(f"[SMB Flag] 파일 없음: {station}, feature={feature}")
        return pd.DataFrame(columns=["날짜"])

    # 2) 파일 로딩
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="cp949")
        except:
            df = pd.read_csv(f, encoding="utf-8")

        # 날짜 컬럼 통일
        if "obs_time" in df.columns:
            df.rename(columns={"obs_time": "날짜"}, inplace=True)
        elif "datetime" in df.columns:
            df.rename(columns={"datetime": "날짜"}, inplace=True)
        elif "癤퓇bs_time" in df.columns:
            df.rename(columns={"癤퓇bs_time": "날짜"}, inplace=True)

        df["날짜"] = pd.to_datetime(df["날짜"])
        
        # 날짜 간격 맞추기
        df = df.sort_values("날짜").reset_index(drop=True)
        if station in ["SMB1", "SMB2", "SMB3", "SMB4"]:
            rule = "10T"   # 10분
        elif station in ["SMB5", "SMB6"]:
            rule = "30T"   # 30분
        else:
            rule = "1T"    # 나머지(Sea_level 등은 1분)

        df = df.set_index("날짜")
        start = df.index.min()
        end = df.index.max()
        new_index = pd.date_range(start=start, end=end, freq=rule)
        df = df.reindex(new_index)
        df = df.reset_index().rename(columns={"index": "날짜"})

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values("날짜").reset_index(drop=True)

    # 3) OBS → FLAG 매핑
    obs_cols = get_columns(station, feature)

    rename_map = {}
    for obs_col in obs_cols:
        if obs_col == "날짜":
            continue
        flag_col = FLAG_COLUMN_MAPPING.get(obs_col)
        if flag_col and flag_col in df_all.columns:
            rename_map[flag_col] = f"FLAG_{obs_col}"

    keep_cols = ["날짜"] + list(rename_map.keys())
    keep_cols = [c for c in keep_cols if c in df_all.columns]

    df_flag = df_all[keep_cols].rename(columns=rename_map)

    # 4) Flag → 정상/이상/결측 매핑
    for col in df_flag.columns:
        if col.startswith("FLAG_"):
            df_flag[col] = pd.to_numeric(df_flag[col], errors="coerce").astype("Int64")
            df_flag[col] = df_flag[col].apply(qc_func)

    return df_flag

#%%
var_map={
    "Repr_Basin Level(E.L.[m])":"Repr_Lake",
    "Exis_Basin Level(E.L.[m])":"Exis_Lake",
    "Repr_Sea Level(E.L.[m])":"Repr_Sea",
    "Exis_Sea Level(E.L.[m])":"Exis_Sea",
    "Tower1(E.L.[m])":"Tower1",
    "Tower2(E.L.[m])":"Tower2"
    }

def sea_level_load_flag_group(base_dir, station, qc_type="qc2"):
    
    base_dir = Path(base_dir)
    qc_func = QC_CONVERTERS.get(qc_type)
    if qc_func is None:
        raise ValueError(f"알 수 없는 QC 타입: {qc_type}")
    
    files = [
        f for f in base_dir.iterdir()
        if f.is_file() and station.lower() in f.name.lower()
    ]

    if not files:
        print(f"[Sea_level FLAG] 파일 없음: {station}")
        return pd.DataFrame(columns=["날짜", f"FLAG_{station}"])

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="cp949")
        except:
            df = pd.read_csv(f, encoding="utf-8")

        if "obs_time" in df.columns:
            df = df.rename(columns={"obs_time": "날짜"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "날짜"})
        elif "Time" in df.columns:
            df = df.rename(columns={"Time": "날짜"})
        
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        
        df = df.sort_values("날짜").reset_index(drop=True)
        df = df.drop_duplicates(subset="날짜", keep="first")
        
        rule = "1T"

        df = df.set_index("날짜")
        start = df.index.min()
        end = df.index.max()
        new_index = pd.date_range(start=start, end=end, freq=rule)
        df = df.reindex(new_index)
        df = df.reset_index().rename(columns={"index": "날짜"})

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True).sort_values("날짜")

    out_df = df_all[["날짜"]].copy()
    
    # 변수 이름 변경
    for raw_var, new_var in var_map.items():
        if raw_var in df_all.columns:
            out_df[new_var] = df_all[raw_var]
        else:
            # raw_var 컬럼이 아예 없으면 NaN 컬럼 생성
            out_df[new_var] = np.nan
    
    # 2QC 데이터 찾아서 출력
    for idx, col in enumerate(df_all.columns):
        col_str = str(col)
    
        # 2QC / 2QC.1 / 2QC.2 ... 만 대상
        if not col_str.startswith("2QC"):
            continue
    
        # 변수 컬럼은 항상 QC 컬럼에서 2칸 앞
        var_idx = idx - 2
        if var_idx < 0:
            continue  # 안전장치
    
        raw_var = df_all.columns[var_idx]  # 예: "Repr_Basin Level(E.L.[m])"
    
        # var_map으로 새 변수명 매핑 (없으면 원래 이름 사용)
        new_var = var_map.get(raw_var, raw_var)
    
        qc_series = df_all[col]
    
        # 데이터 하나도 없으면 패스 (전부 NaN)
        if qc_series.dropna().empty:
            print(f"[Sea_level FLAG] QC2 데이터 없음(전부 NaN): 변수={raw_var}, qc컬럼={col_str}")
            out_df[f"FLAG_{new_var}"] = np.nan
            continue
    
        # QC 변환 적용 (NaN은 그대로 유지)
        out_df[f"FLAG_{new_var}"] = qc_series.apply(
            lambda x: qc_func(x) if pd.notna(x) else np.nan
        )

    return out_df

def sea_level_load_1qc_flag_group(base_dir, station, qc_type="qc1"):
    
    base_dir = Path(base_dir)
    qc_func = QC_CONVERTERS.get(qc_type)
    if qc_func is None:
        raise ValueError(f"알 수 없는 QC 타입: {qc_type}")
    
    files = [
        f for f in base_dir.iterdir()
        if f.is_file() and station.lower() in f.name.lower()
    ]

    if not files:
        print(f"[Sea_level FLAG] 파일 없음: {station}")
        return pd.DataFrame(columns=["날짜", f"FLAG_{station}"])

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="cp949")
        except:
            df = pd.read_csv(f, encoding="utf-8")

        if "obs_time" in df.columns:
            df = df.rename(columns={"obs_time": "날짜"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "날짜"})
        elif "Time" in df.columns:
            df = df.rename(columns={"Time": "날짜"})
        
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        
        df = df.sort_values("날짜").reset_index(drop=True)
        df = df.drop_duplicates(subset="날짜", keep="first")
        
        rule = "1T"

        df = df.set_index("날짜")
        start = df.index.min()
        end = df.index.max()
        new_index = pd.date_range(start=start, end=end, freq=rule)
        df = df.reindex(new_index)
        df = df.reset_index().rename(columns={"index": "날짜"})

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True).sort_values("날짜")
    df_out = df_all[["날짜", "waterlevel_qc_result"]]

    df_out["waterlevel_qc_result"] = df_all["waterlevel_qc_result"].apply(
        lambda x: qc_func(x) if pd.notna(x) else np.nan
    )
    return df_out

def merge_data_and_flags(obs_df, flag_df, station):

    # ========== 1) Train 기간 정의 ==========
    if station in station_periods:
        start_date, train_end_date = station_periods[station]
        start_date = pd.to_datetime(start_date)
        train_end_date = pd.to_datetime(train_end_date)
    else:
        raise ValueError(f"Unknown station: {station}")

    # ========== 2) 날짜 정렬 ==========
    obs_df = obs_df.sort_values("날짜")
    flag_df = flag_df.sort_values("날짜")

    # ========== 3) 병합 수행 ==========
    merged = pd.merge(obs_df, flag_df, on="날짜", how="left")

    # ========== 4) Train/Test 분리 ==========
    train_df = merged[(merged["날짜"] >= start_date) & (merged["날짜"] <= train_end_date)].copy()
    test_df  = merged[merged["날짜"] > train_end_date].copy()

    return train_df, test_df

def flags_split(flag_df, station):

    # ========== 1) Train 기간 정의 ==========
    if station in station_periods:
        start_date, train_end_date = station_periods[station]
        start_date = pd.to_datetime(start_date)
        train_end_date = pd.to_datetime(train_end_date)
    else:
        raise ValueError(f"Unknown station: {station}")

    # ========== 2) 날짜 정렬 ==========
    flag_df = flag_df.sort_values("날짜")

    # ========== 4) Train/Test 분리 ==========
    train_df = flag_df[(flag_df["날짜"] >= start_date) & (flag_df["날짜"] <= train_end_date)].copy()
    test_df  = flag_df[flag_df["날짜"] > train_end_date].copy()

    return train_df, test_df

def convert_dir_to_uv(df):
    df_new = df.copy()
    final_order = []

    for col in df.columns:

        # "DIR" 또는 한글 "향" 포함 컬럼
        is_dir = ('DIR' in col.upper()) or ('향' in col)

        if is_dir:
            try:
                direction_rad = np.deg2rad(df_new[col])

                # U/V 컬럼명 만들기 (원본명 그대로 활용)
                # 예: 풍향(deg) → 풍향_U, 풍향_V
                base_name = col.replace("(deg)", "").replace("(DEG)", "").replace("(Deg)", "")
                base_name = base_name.strip()

                u_col = f"{base_name}_U"
                v_col = f"{base_name}_V"

                df_new[u_col] = -np.sin(direction_rad)
                df_new[v_col] = -np.cos(direction_rad)

                final_order.extend([u_col, v_col])

                # 원본 DIR/향 컬럼 삭제
                df_new.drop(columns=[col], inplace=True)

            except Exception as e:
                print(f"[경고] '{col}' 변환 실패: {e}")
                final_order.append(col)

        else:
            final_order.append(col)


    # 최종 컬럼 순서 정리
    return df_new[final_order]

def convert_uv_to_dir(df):
    df_new = df.copy()
    
    # '_U'로 끝나는 모든 열 찾기
    u_cols = [col for col in df_new.columns if col.endswith('_U')]
    
    for u_col in u_cols:
        # 접두어 추출 (예: 'WIND_U' -> 'WIND')
        prefix = u_col[:-2]
        v_col = prefix + '_V'
        
        if v_col in df_new.columns:
            # U, V로부터 방향 계산 (풍향이 'from'의 방향)
            df_new[prefix + '_DIR'] = (np.degrees(np.arctan2(-df_new[u_col], -df_new[v_col])) % 360)
        else:
            print(f"Warning: '{v_col}' column not found for '{u_col}'. Skipping conversion for this pair.")
        
        # 기존 U, V 열 제거
        df_new.drop(columns=[u_col, v_col], inplace=True)
    
    return df_new

def convert_flag_dir_to_uv(flag_df, time_col="날짜"):
    df_new = flag_df.copy()
    final_order = []

    for col in flag_df.columns:

        # 날짜는 그대로
        if col == time_col:
            final_order.append(col)
            continue

        # DIR 또는 "향"이 들어가는 FLAG 컬럼 찾기
        is_dir_flag = ("DIR" in col.upper()) or ("향" in col)

        if is_dir_flag:
            try:
                # base_name 만드는 방식 (원래 convert_dir_to_uv와 동일)
                base_name = col.replace("(deg)", "").replace("(DEG)", "").replace("(Deg)", "")
                base_name = base_name.strip()

                # FLAG_DIR → FLAG_DIR_U / FLAG_DIR_V
                u_col = f"{base_name}_U"
                v_col = f"{base_name}_V"

                # 값은 그대로 복사 (변환 없음)
                df_new[u_col] = df_new[col]
                df_new[v_col] = df_new[col]

                # 새 컬럼 순서 저장
                final_order.extend([u_col, v_col])

                # 원본 DIR flag 삭제
                df_new.drop(columns=[col], inplace=True)

            except Exception as e:
                print(f"[FLAG 변환 실패] '{col}': {e}")
                final_order.append(col)
        else:
            # 일반 flag는 그대로 보존
            final_order.append(col)

    # 최종 컬럼 순서 적용
    df_new = df_new[final_order]

    return df_new

#%% 그림으로 qc된 데이터확인하기
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 현재까지의 스타일 설정을 초기화
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 표시 문제를 해결
try:
    local_font = BASE_DIR / "assets" / "fonts" / "NanumGothic.ttf"
    if local_font.exists():
        font = fm.FontProperties(fname=str(local_font))
        plt.rc('font', family=font.get_name())
except Exception:
    pass


def build_flag_pairs(df):

    exclude = {"날짜", "year", "month"}
    cols = set(df.columns) - exclude

    flag_cols = [c for c in cols if c.startswith("FLAG_")]
    non_flag_cols = [c for c in cols if not c.startswith("FLAG_")]

    pairs = []

    for flag in flag_cols:
        base = flag.replace("FLAG_", "")
        if base in non_flag_cols:
            pairs.append((base, flag))

    return pairs

def plot_with_flags(df_):
    df = df_.copy()
    variables = build_flag_pairs(df)
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3*n), sharex=True)
    
    for ax, (var, flag) in zip(axes, variables):

        # --- 기본 라인 ---
        ax.plot(df["날짜"], df[var], marker='o', markersize=3, linestyle='-', label=var, zorder=1)

        # --- FLAG==1 인 지점에 빨간 X 표시 ---
        mask = df[flag] == 1
        ax.scatter(df.loc[mask, "날짜"], df.loc[mask, var],
                   marker='x', s=60, color='red', label='FLAG=1', zorder=2)

        ax.set_ylabel(var)
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel("날짜")
    plt.tight_layout()
    plt.show()

def plot_with_flags_by_year(df_):
    df = df_.copy()
    df["year"] = df["날짜"].dt.year

    years = sorted(df["year"].unique())

    for y in years:
        df_y = df[df["year"] == y]
        if df_y.empty:
            continue

        print(f"\n===== {y}년 데이터 시각화 =====")
        plot_with_flags(df_y)

#%%

