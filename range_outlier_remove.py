# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:57:44 2025

@author: user
"""
import re
import pandas as pd
import numpy as np

#%%

def apply_physical_limits(df):

    physical_limits = {
        "풍향(deg)":(0,360),
        "풍속(m/s)":(0.01, 40), 
        "최대풍속(m/s)":(0.01, 40),
        "기온(℃)":(-40, 55),
        "기압(hPa)":(800, 1080),
        "일사(W/m2)":(-9, 2000),
        "유향(deg)":(0,360),
        "유속(Cm/s)":(0.01, 300),
        "수온(℃)":(-2, 40),
        "탁도(NTU)":(0.01, 1000), 
        "염분(PSU)":(0.01, 40),
        "O2(%)":(0.01, 500),
        "O2(ppm)":(0.01, 45), 
        "pH":(6, 11),
        "chlorophyll":(0.01, 1000),
        "유의파고(m)":(0.01, 8),
        "파주기(s)":(0.01, 20),
        "파향(deg)":(0,360)
    }
    df_new = df.copy()

    # 정규화된 물리한계 key 리스트 만들기
    norm_limits = {}
    for key, val in physical_limits.items():
        norm_key = re.sub(r"[\s\(\)]", "", key).lower()   # 공백/괄호 제거 + 소문자화
        norm_limits[norm_key] = val

    for col in df.columns:
        if col == "날짜":
            continue

        # 1) 변수명 추출 ("SMB1_기압(hPa)" → "기압(hPa)")
        try:
            raw_key = col.split("_", 1)[1]
        except:
            continue

        # 2) 정규화 (괄호/공백 제거 + 소문자)
        norm_key = re.sub(r"[\s\(\)]", "", raw_key).lower()

        # 3) 물리 한계 대응되는 key 찾기
        matched_key = None
        for k in norm_limits.keys():
            if k in norm_key:    # 부분 포함 match
                matched_key = k
                break

        # 물리 한계에 없는 변수면 skip
        if matched_key is None:
            continue

        min_val, max_val = norm_limits[matched_key]

        # float 변환
        vals = pd.to_numeric(df_new[col], errors="coerce")

        # 범위 벗어난 값 NaN
        outlier_mask = (vals < min_val) | (vals > max_val)
        df_new.loc[outlier_mask, col] = np.nan

    return df_new

