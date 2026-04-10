#%%
import os
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
PROJECT_ROOT = Path(BASE_DIR).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FOR_AI_DIR = PROJECT_ROOT / "data_for_AI"
DATA_FOR_AI_DF_UNI_DIR = DATA_FOR_AI_DIR / "df_univariate"
DATA_FOR_AI_DF_DIR = DATA_FOR_AI_DIR / "df"
from df_utils import (
    load_obs_group, 
    smb_load_flag_group,
    sea_level_load_flag_group,
    merge_data_and_flags,
    sea_level_load_1qc_flag_group,
    flags_split,
    plot_with_flags_by_year)
from df_utils import convert_dir_to_uv, convert_uv_to_dir
from functools import reduce

import pandas as pd
import inspect
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ?꾩옱源뚯????ㅽ????ㅼ젙??珥덇린??
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False # 留덉씠?덉뒪 遺???쒖떆 臾몄젣瑜??닿껐
try:
    local_font = Path(BASE_DIR) / "assets" / "fonts" / "NanumGothic.ttf"
    if local_font.exists():
        font = fm.FontProperties(fname=str(local_font))
        plt.rc('font', family=font.get_name())
except Exception:
    pass

import json

def load_station_periods(json_path=None):
    if json_path is None:
        json_path = os.path.join(BASE_DIR, "json_folder", "station_periods.json")
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 臾몄옄?댁쓣 datetime ?뺥깭濡?蹂?섑빐二쇰뒗 ?뺥깭濡??뺣━
    periods = {
        k: (pd.to_datetime(v[0]), pd.to_datetime(v[1]))
        for k, v in data.items()
    }
    return periods

# 濡쒕뵫
station_periods = load_station_periods()

#%% 媛꾨떒 ?뚯뒪?몄슜 main
# if __name__ == "__main__":
base_dir = str(DATA_DIR / "1_OBS")
QC1_FLAG_DIR = str(DATA_DIR / "10_FLAG1")
QC2_FLAG_DIR = str(DATA_DIR / "20_FLAG2")

# SMB1, SMB2 -> Air, Surface
# SMB3, SMB4 -> Surface, Water 
# SMB5, SMB6 -> None
# Sea_level -> None (qc2媛 ?놁쓬)

#%% ?곗씠??遺덈윭?ㅺ린
print("?뜯뼳??Loading OBS and Flag data...")

# SMB1, SMB2 -> Air, Surface
SMB1_Air_obs = load_obs_group(base_dir, station="SMB1", feature="Air")
SMB1_Air_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB1", feature="Air", qc_type="qc2")
SMB1_Air_train, SMB1_Air_test = merge_data_and_flags(SMB1_Air_obs, SMB1_Air_qc, station="SMB1")

SMB2_Air_obs = load_obs_group(base_dir, station="SMB2", feature="Air")
SMB2_Air_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB2", feature="Air", qc_type="qc2")
SMB2_Air_train, SMB2_Air_test = merge_data_and_flags(SMB2_Air_obs, SMB2_Air_qc, station="SMB2")

SMB1_Surface_obs = load_obs_group(base_dir, station="SMB1", feature="Surface")
SMB1_Surface_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB1", feature="Surface", qc_type="qc2")
SMB1_Surface_train, SMB1_Surface_test = merge_data_and_flags(SMB1_Surface_obs, SMB1_Surface_qc, station="SMB1")

SMB2_Surface_obs = load_obs_group(base_dir, station="SMB2", feature="Surface")
SMB2_Surface_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB2", feature="Surface", qc_type="qc2")
SMB2_Surface_train, SMB2_Surface_test = merge_data_and_flags(SMB2_Surface_obs, SMB2_Surface_qc, station="SMB2")

# SMB3, SMB4 -> Surface, Water 
SMB3_Surface_obs = load_obs_group(base_dir, station="SMB3", feature="Surface")
SMB3_Surface_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB3", feature="Surface", qc_type="qc2")
SMB3_Surface_train, SMB3_Surface_test = merge_data_and_flags(SMB3_Surface_obs, SMB3_Surface_qc, station="SMB3")

SMB4_Surface_obs = load_obs_group(base_dir, station="SMB4", feature="Surface")
SMB4_Surface_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB4", feature="Surface", qc_type="qc2")
SMB4_Surface_train, SMB4_Surface_test = merge_data_and_flags(SMB4_Surface_obs, SMB4_Surface_qc, station="SMB4")

SMB3_Water_obs = load_obs_group(base_dir, station="SMB3", feature="Water")
SMB3_Water_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB3", feature="Water", qc_type="qc2")
SMB3_Water_train, SMB3_Water_test = merge_data_and_flags(SMB3_Water_obs, SMB3_Water_qc, station="SMB3")

SMB4_Water_obs = load_obs_group(base_dir, station="SMB4", feature="Water")
SMB4_Water_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB4", feature="Water", qc_type="qc2")
SMB4_Water_train, SMB4_Water_test = merge_data_and_flags(SMB4_Water_obs, SMB4_Water_qc, station="SMB4")

# SMB5, SMB6 -> None
SMB5_obs = load_obs_group(base_dir, station="SMB5", feature=None)
SMB5_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB5", feature=None, qc_type="qc2")
SMB5_train, SMB5_test = merge_data_and_flags(SMB5_obs, SMB5_qc, station="SMB5")

SMB6_obs = load_obs_group(base_dir, station="SMB6", feature=None)
SMB6_qc = smb_load_flag_group(QC2_FLAG_DIR, station="SMB6", feature=None, qc_type="qc2")
SMB6_train, SMB6_test = merge_data_and_flags(SMB6_obs, SMB6_qc, station="SMB6")

# Sea_level
Sea_level_obs = load_obs_group(base_dir, station="Sea_level", feature=None)
Sea_level = Sea_level_obs.copy()

sbasin_flag = sea_level_load_flag_group(QC2_FLAG_DIR, "Sihwa_Basin", qc_type="qc2")
stide_flag = sea_level_load_flag_group(QC2_FLAG_DIR, "Sihwa_Tide", qc_type="qc2")
stower_flag = sea_level_load_flag_group(QC2_FLAG_DIR, "Sihwa_Tower", qc_type="qc2")

def merge_flag_columns(Sea_level, flag_df):

    flag_cols = [c for c in flag_df.columns if c.startswith("FLAG_")]
    print(flag_cols)
    merged = Sea_level.merge(flag_df[["?좎쭨"] + flag_cols], on="?좎쭨", how="left")

    return merged

Sea_level = merge_flag_columns(Sea_level, sbasin_flag)
Sea_level = merge_flag_columns(Sea_level, stide_flag)
Sea_level = merge_flag_columns(Sea_level, stower_flag)

start_date, train_end_date = station_periods["Sea_level"]

Sea_level_train = Sea_level[(Sea_level["?좎쭨"] >= pd.to_datetime(start_date)) & (Sea_level["?좎쭨"] <= pd.to_datetime(train_end_date))]
Sea_level_test = Sea_level[(Sea_level["?좎쭨"] > pd.to_datetime(train_end_date))]


#%% ?숈뒿???곗씠?곕겮由?臾띔린
AIR_GROUPS = {
    "wind_speed": ["?띿냽(m/s)"],
    "wind_max_speed":["理쒕??띿냽(m/s)"],
    "temp": ["湲곗삩(??"],
    "solar":["?쇱궗(W/m2)"],
    "wind_dir": ["?랁뼢(deg)"],
    "press": ["湲곗븬(hPa)"]}

SURFACE_GROUPS = {
    "current_dir": ["?좏뼢(deg)"],
    "current_speed": ["?좎냽(Cm/s)"]}

WATER_GROUPS = {
    "temp": ["?섏삩(??"],
    "Salinity": ["?쇰텇(PSU)"],
    "O2Per":["O2(%)"],
    "O2ppm": ["O2(ppm)"],
    "pH":["pH"],
    "chl-a":["chlorophyll"],
    "turbidity":["?곷룄(NTU)"]}

WAVE_GROUPS = {
    "temp": ["?섏삩(??"],
    "wave_H": ["?좎쓽?뚭퀬(m)"],
    "wave_F": ["?뚯＜湲?s)"],
    "wave_dir": ["?뚰뼢(deg)"]}

SEA_LEVEL_GROUPS ={
    "Sea": ["Repr_Sea", "Exis_Sea"],
    "Lake": ["Repr_Lake", "Exis_Lake"],
    "Tower": ["Tower1", "Tower2"]}

#%%
print("?뜯뼳??Group splitting and save data...")
import pandas as pd
from functools import reduce

class GroupSplitter:
    def __init__(self, groups: dict, time_col="?좎쭨"):
        self.groups = groups
        self.time_col = time_col

    def _extract_and_prefix(self, df, cols, prefix):
        val_cols = [c for c in cols if c in df.columns]
        flag_cols = [f"FLAG_{c}" for c in cols if f"FLAG_{c}" in df.columns]

        val_df = df[[self.time_col] + val_cols].copy()
        val_df = val_df.rename(columns={c: f"{prefix}_{c}" for c in val_cols})

        if flag_cols:
            flag_df = df[[self.time_col] + flag_cols].copy()
            flag_df = flag_df.rename(columns={c: f"{prefix}_{c}" for c in flag_cols})
        else:
            flag_df = df[[self.time_col]].copy()

        return val_df, flag_df

    def _merge_dfs(self, dfs):
        if len(dfs) == 1:
            return dfs[0]
        return reduce(
            lambda l, r: pd.merge(l, r, on=self.time_col, how="outer"),
            dfs
        ).sort_values(self.time_col)

    def split(self, *dfs, stations=None):
        assert stations is None or len(stations) == len(dfs)

        if stations is None:
            stations = [f"ST{i+1}" for i in range(len(dfs))]

        value_dfs = {k: [] for k in self.groups.keys()}
        flag_dfs  = {k: [] for k in self.groups.keys()}

        for group_key, cols in self.groups.items():
            each_vals = []
            each_flags = []

            for df, st in zip(dfs, stations):
                val_df, flag_df = self._extract_and_prefix(df, cols, st)
                each_vals.append(val_df)
                each_flags.append(flag_df)

            value_dfs[group_key] = self._merge_dfs(each_vals)
            flag_dfs[group_key]  = self._merge_dfs(each_flags)

        return value_dfs, flag_dfs

    # ---------------------------
    # ?????湲곕뒫 異붽?
    # ---------------------------
    def save(self, value_dfs, flag_dfs, save_dir, file_format="csv"):
        os.makedirs(save_dir, exist_ok=True)

        for group_key in value_dfs.keys():

            val_df = value_dfs[group_key]
            flag_df = flag_dfs[group_key]

            if file_format == "csv":
                val_path  = os.path.join(save_dir, f"{group_key}_values.csv")
                flag_path = os.path.join(save_dir, f"{group_key}_flags.csv")
                val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
                flag_df.to_csv(flag_path, index=False, encoding="utf-8-sig")

            elif file_format == "parquet":
                val_path  = os.path.join(save_dir, f"{group_key}_values.parquet")
                flag_path = os.path.join(save_dir, f"{group_key}_flags.parquet")
                val_df.to_parquet(val_path, index=False)
                flag_df.to_parquet(flag_path, index=False)

            else:
                raise ValueError("吏?먰븯吏 ?딅뒗 ?뚯씪 ?뺤떇?낅땲??")

def save_group_split(groups, train_dfs, test_dfs, stations, base_save_dir):
    """
    groups: AIR_GROUPS 媛숈? dict
    train_dfs: [SMB1_Air_train, SMB2_Air_train] 媛숈? 由ъ뒪??
    test_dfs : [SMB1_Air_test, SMB2_Air_test]
    stations: ["SMB1","SMB2"]
    base_save_dir: "<project_root>/data_for_AI/df_univariate/SMB1_AIR"
    """

    # ??splitter ?앹꽦
    splitter = GroupSplitter(groups)

    # ??train 泥섎━
    train_vals, train_flags = splitter.split(
        *train_dfs,
        stations=stations
    )
    train_save_dir = base_save_dir + "_train"
    splitter.save(train_vals, train_flags, save_dir=train_save_dir)

    print(f"[?꾨즺] Train ?????{train_save_dir}")

    # ??test 泥섎━
    test_vals, test_flags = splitter.split(
        *test_dfs,
        stations=stations
    )
    test_save_dir = base_save_dir + "_test"
    splitter.save(test_vals, test_flags, save_dir=test_save_dir)

    print(f"[?꾨즺] Test ?????{test_save_dir}")
    
#%% SMB1_SBM2_AIR
save_group_split(
    groups=AIR_GROUPS,
    train_dfs=[SMB1_Air_train],
    test_dfs=[SMB1_Air_test],
    stations=["SMB1"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB1_AIR'))

save_group_split(
    groups=AIR_GROUPS,
    train_dfs=[SMB2_Air_train],
    test_dfs=[SMB2_Air_test],
    stations=["SMB2"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB2_AIR'))

#%% SMB1_SBM2_SURFACE
save_group_split(
    groups=SURFACE_GROUPS,
    train_dfs=[SMB1_Surface_train],
    test_dfs=[SMB1_Surface_test],
    stations=["SMB1"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB1_SURFACE'))

save_group_split(
    groups=SURFACE_GROUPS,
    train_dfs=[SMB2_Surface_train],
    test_dfs=[SMB2_Surface_test],
    stations=["SMB2"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB2_SURFACE'))

#%% SMB3_SBM4_SURFACE
save_group_split(
    groups=SURFACE_GROUPS,
    train_dfs=[SMB3_Surface_train],
    test_dfs=[SMB3_Surface_test],
    stations=["SMB3"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB3_SURFACE'))

save_group_split(
    groups=SURFACE_GROUPS,
    train_dfs=[SMB4_Surface_train],
    test_dfs=[SMB4_Surface_test],
    stations=["SMB4"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB4_SURFACE'))

#%% SMB3_SBM4_WATER
save_group_split(
    groups=WATER_GROUPS,
    train_dfs=[SMB3_Water_train],
    test_dfs=[SMB3_Water_test],
    stations=["SMB3"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB3_WATER'))

save_group_split(
    groups=WATER_GROUPS,
    train_dfs=[SMB4_Water_train],
    test_dfs=[SMB4_Water_test],
    stations=["SMB4"],
    base_save_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB4_WATER'))


#%% SEA_LEVEL
save_group_split(
    groups=SEA_LEVEL_GROUPS,
    train_dfs=[Sea_level_train],
    test_dfs=[Sea_level_test],
    stations=["Sea_level"],
    base_save_dir=str(DATA_FOR_AI_DF_DIR / 'Sea_level'))

#%%

SMB1_Air_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB1", feature="Air", qc_type="qc1")
SMB2_Air_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB2", feature="Air", qc_type="qc1")
SMB1_Surface_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB1", feature="Surface", qc_type="qc1")
SMB2_Surface_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB2", feature="Surface", qc_type="qc1")
SMB3_Surface_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB3", feature="Surface", qc_type="qc1")
SMB4_Surface_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB4", feature="Surface", qc_type="qc1")
SMB3_Water_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB3", feature="Water", qc_type="qc1")
SMB4_Water_qc1 = smb_load_flag_group(QC1_FLAG_DIR, station="SMB4", feature="Water", qc_type="qc1")
lake_flag_qc1 = sea_level_load_1qc_flag_group(QC1_FLAG_DIR, "Repr_Lake", qc_type="qc1")
sea_flag_qc1 = sea_level_load_1qc_flag_group(QC1_FLAG_DIR, "Repr_Sea", qc_type="qc1")



SMB1_Air_qc1_train, SMB1_Air_qc1_test = flags_split(SMB1_Air_qc1, station="SMB1")
SMB2_Air_qc1_train, SMB2_Air_qc1_test = flags_split(SMB2_Air_qc1, station="SMB2")
SMB1_Surface_qc1_train, SMB1_Surface_qc1_test = flags_split(SMB1_Surface_qc1, station="SMB1")
SMB2_Surface_qc1_train, SMB2_Surface_qc1_test = flags_split(SMB2_Surface_qc1, station="SMB2")
SMB3_Surface_qc1_train, SMB3_Surface_qc1_test = flags_split(SMB3_Surface_qc1, station="SMB3")
SMB4_Surface_qc1_train, SMB4_Surface_qc1_test = flags_split(SMB4_Surface_qc1, station="SMB4")
SMB3_Water_qc1_train, SMB3_Water_qc1_test = flags_split(SMB3_Water_qc1, station="SMB3")
SMB4_Water_qc1_train, SMB4_Water_qc1_test = flags_split(SMB4_Water_qc1, station="SMB4")
lake_flag_qc1_train, lake_flag_qc1_test = flags_split(lake_flag_qc1, station="Sea_level")
sea_flag_qc1_train, sea_flag_qc1_test = flags_split(sea_flag_qc1, station="Sea_level")

lake_flag_qc1_train.to_csv(str(DATA_FOR_AI_DF_DIR / 'Sea_level_train' / 'Repr_Lake_flags_1qc.csv'), encoding='cp949', index=False)
sea_flag_qc1_train.to_csv(str(DATA_FOR_AI_DF_DIR / 'Sea_level_train' / 'Repr_Sea_flags_1qc.csv'), encoding='cp949', index=False)
lake_flag_qc1_test.to_csv(str(DATA_FOR_AI_DF_DIR / 'Sea_level_test' / 'Repr_Lake_flags_1qc.csv'), encoding='cp949', index=False)
sea_flag_qc1_test.to_csv(str(DATA_FOR_AI_DF_DIR / 'Sea_level_test' / 'Repr_Sea_flags_1qc.csv'), encoding='cp949', index=False)

#%%
import os
import pandas as pd
from functools import reduce

# ======================================================
# ??FLAG ?꾩슜 Splitter
# ======================================================
class FlagOnlySplitter:
    def __init__(self, groups: dict, time_col="?좎쭨"):
        self.groups = groups
        self.time_col = time_col

    # ---------------------------
    # 1) FLAG留?異붿텧 + prefix
    # ---------------------------
    def _extract_flag_only(self, df, cols, prefix):
        """
        df?먯꽌 value ?쒖쇅, FLAG 而щ읆留?異붿텧 ??prefix 遺??
        FLAG_xxx -> ST1_FLAG_xxx ?뺥깭
        """
        # FLAG 議댁옱?섎뒗 而щ읆留??꾪꽣留?
        flag_cols = [f"FLAG_{c}" for c in cols if f"FLAG_{c}" in df.columns]

        # FLAG媛 ?놁쑝硫??좎쭨留?諛섑솚
        if not flag_cols:
            return df[[self.time_col]].copy()

        # ?좎쭨 + FLAG留?異붿텧
        out = df[[self.time_col] + flag_cols].copy()

        # prefix 遺??
        out = out.rename(columns={c: f"{prefix}_{c}" for c in flag_cols})
        return out

    # ---------------------------
    # 2) ?щ윭 DF 蹂묓빀 (outer join)
    # ---------------------------
    def _merge(self, dfs):
        if len(dfs) == 1:
            return dfs[0]
        return reduce(
            lambda l, r: pd.merge(l, r, on=self.time_col, how="outer"),
            dfs
        ).sort_values(self.time_col)

    # ---------------------------
    # 3) FLAG留?split ?섑뻾
    # ---------------------------
    def split(self, *dfs, stations=None):
        """
        FLAG留?group 湲곗??쇰줈 遺꾨━?섏뿬 諛섑솚
        return: {group_key : FLAG_df}
        """
        assert stations is None or len(stations) == len(dfs)

        if stations is None:
            stations = [f"ST{i+1}" for i in range(len(dfs))]

        result = {g: [] for g in self.groups.keys()}

        for group_key, cols in self.groups.items():
            temp_list = []

            for df, st in zip(dfs, stations):
                flag_df = self._extract_flag_only(df, cols, st)
                temp_list.append(flag_df)

            merged = self._merge(temp_list)
            result[group_key] = merged

        return result

    # ---------------------------
    # 4) FLAG留????
    # ---------------------------
    def save(self, flag_dict, save_dir, file_format="csv"):
        os.makedirs(save_dir, exist_ok=True)

        for group_key, df in flag_dict.items():
            if file_format == "csv":
                path = os.path.join(save_dir, f"{group_key}_flags_1qc.csv")
                df.to_csv(path, index=False, encoding="utf-8-sig")

            elif file_format == "parquet":
                path = os.path.join(save_dir, f"{group_key}_flags_qc.parquet")
                df.to_parquet(path, index=False)

            else:
                raise ValueError("吏?먰븯吏 ?딅뒗 ?뚯씪 ?뺤떇?낅땲??")


# ======================================================
# ??理쒖쥌 ?몄텧 ?⑥닔: FLAG留?遺꾨━ ???
# ======================================================
def save_flag_only_groups(groups, train_flag_dfs, test_flag_dfs, stations, base_dir):

    splitter = FlagOnlySplitter(groups)

    # ---- Train ----
    train_flags = splitter.split(*train_flag_dfs, stations=stations)
    train_dir = base_dir + "_train"
    splitter.save(train_flags, save_dir=train_dir)
    print(f"[?꾨즺] Train FLAG ?????{train_dir}")

    # ---- Test ----
    test_flags = splitter.split(*test_flag_dfs, stations=stations)
    test_dir = base_dir + "_test"
    splitter.save(test_flags, save_dir=test_dir)
    print(f"[?꾨즺] Test FLAG ?????{test_dir}")

#%%

save_flag_only_groups(
    groups=AIR_GROUPS,
    train_flag_dfs=[SMB1_Air_qc1_train],
    test_flag_dfs=[SMB1_Air_qc1_test],
    stations=["SMB1"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB1_AIR'))

save_flag_only_groups(
    groups=AIR_GROUPS,
    train_flag_dfs=[SMB2_Air_qc1_train],
    test_flag_dfs=[SMB2_Air_qc1_test],
    stations=["SMB2"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB2_AIR'))

save_flag_only_groups(
    groups=SURFACE_GROUPS,
    train_flag_dfs=[SMB1_Surface_qc1_train],
    test_flag_dfs=[SMB1_Surface_qc1_test],
    stations=["SMB1"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB1_SURFACE'))

save_flag_only_groups(
    groups=SURFACE_GROUPS,
    train_flag_dfs=[SMB2_Surface_qc1_train],
    test_flag_dfs=[SMB2_Surface_qc1_test],
    stations=["SMB2"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB2_SURFACE'))

save_flag_only_groups(
    groups=SURFACE_GROUPS,
    train_flag_dfs=[SMB3_Surface_qc1_train],
    test_flag_dfs=[SMB3_Surface_qc1_test],
    stations=["SMB3"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB3_SURFACE'))

save_flag_only_groups(
    groups=SURFACE_GROUPS,
    train_flag_dfs=[SMB4_Surface_qc1_train],
    test_flag_dfs=[SMB4_Surface_qc1_test],
    stations=["SMB4"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB4_SURFACE'))

save_flag_only_groups(
    groups=WATER_GROUPS,
    train_flag_dfs=[SMB3_Water_qc1_train],
    test_flag_dfs=[SMB3_Water_qc1_test],
    stations=["SMB3"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB3_WATER'))

save_flag_only_groups(
    groups=WATER_GROUPS,
    train_flag_dfs=[SMB4_Water_qc1_train],
    test_flag_dfs=[SMB4_Water_qc1_test],
    stations=["SMB4"],
    base_dir=str(DATA_FOR_AI_DF_UNI_DIR / 'SMB4_WATER'))







