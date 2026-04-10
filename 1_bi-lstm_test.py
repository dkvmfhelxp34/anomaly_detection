# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 17:36:57 2025

@author: user
"""

# -*- coding: utf-8 -*-
"""
BiLSTM-AE
"""

import os
import re
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
PROJECT_ROOT = Path(BASE_DIR).parent
DATA_FOR_AI_DIR = PROJECT_ROOT / "data_for_AI"
OUTPUT_DIR = PROJECT_ROOT / "output"
import random
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from df_utils import convert_dir_to_uv, convert_flag_dir_to_uv
from range_outlier_remove import apply_physical_limits

# torch
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Wrappers
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
seed_everything(42)

def list_pairs(data_dir):
    """
    data_dir 안에 있는 *_values.csv, *_flags.csv 쌍을 찾아 pair 리스트로 반환
    """
    files = os.listdir(data_dir)
    values = [f.replace("_values.csv", "") for f in files if f.endswith("_values.csv")]
    flags  = [f.replace("_flags.csv", "") for f in files if f.endswith("_flags.csv")]

    # values 와 flags 둘 다 있는 pair만 선택
    pairs = sorted(list(set(values) & set(flags)))
    return pairs

def load_selected_pair(train_dir, test_dir, pair_name):
    """
    train_dir/test_dir에서 동일 pair_name의
    values/flags를 모두 불러서 4개의 DF를 반환한다.

    반환:
        train_values_df, train_flags_df,
        test_values_df,  test_flags_df
    """

    # -------------------------
    # Train 파일 경로
    # -------------------------
    train_values_path = os.path.join(train_dir, f"{pair_name}_values.csv")
    train_flags_path  = os.path.join(train_dir, f"{pair_name}_flags.csv")

    # -------------------------
    # Test 파일 경로
    # -------------------------
    test_values_path = os.path.join(test_dir, f"{pair_name}_values.csv")
    test_flags_path  = os.path.join(test_dir, f"{pair_name}_flags.csv")

    # -------------------------
    # 파일 존재 여부 확인
    # -------------------------
    for p in [train_values_path, train_flags_path, test_values_path, test_flags_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"파일 없음: {p}")

    # -------------------------
    # Load CSVs
    # -------------------------
    def read_csv_safe(path):
        try:
            return pd.read_csv(path, encoding="cp949")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
        
    train_values_df = read_csv_safe(train_values_path)
    train_flags_df  = read_csv_safe(train_flags_path)

    test_values_df  = read_csv_safe(test_values_path)
    test_flags_df   = read_csv_safe(test_flags_path)

    # 날짜 정렬 (있으면)
    if "날짜" in train_values_df.columns:
        train_values_df = train_values_df.sort_values("날짜")
    if "날짜" in train_flags_df.columns:
        train_flags_df = train_flags_df.sort_values("날짜")

    if "날짜" in test_values_df.columns:
        test_values_df = test_values_df.sort_values("날짜")
    if "날짜" in test_flags_df.columns:
        test_flags_df = test_flags_df.sort_values("날짜")

    return train_values_df, train_flags_df, test_values_df, test_flags_df


class AIDataBuilder:
    def __init__(self, seq_len=20, save_dir=None, time_col="날짜", val_ratio=0.2, random_state=42, apply_flag_nan=False, use_normalize=True):
        self.seq_len = seq_len
        self.time_col = time_col
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.apply_flag_nan = apply_flag_nan
        self.use_normalize = use_normalize
        self.scaler = None
        self.save_dir = save_dir
        
        # 저장 폴더 없으면 생성
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
    
    # -------------------------------------------------------
    # 1. range QC 진행
    # -------------------------------------------------------
    def apply_physical_limits(self, df):
        return apply_physical_limits(df) 
    
    def apply_flag_qc(self, train_values, train_flags, time_col="날짜"):
        values_nan = train_values.copy()
    
        # 날짜를 제외한 컬럼
        value_cols = [c for c in train_values.columns if c != time_col]
        flag_cols  = [c for c in train_flags.columns if c != time_col]
    
        # -------- 1) value 컬럼 정규화 --------
        def normalize_value_name(name):
            # '_' 제거
            return name.replace("_", "")
    
        # -------- 2) flag 컬럼 정규화 --------
        def normalize_flag_name(name):
            # FLAG 관련 패턴 제거
            name = re.sub(r'FLAG', '', name)
            # '_' 제거
            name = name.replace("_", "")
            return name
    
        # FLAG 정규화 → 원본 컬럼 매핑
        normalized_flag = {normalize_flag_name(fc): fc for fc in flag_cols}
    
        # 매칭 dictionary
        col_map = {}
    
        for vcol in value_cols:
            norm_v = normalize_value_name(vcol)
            if norm_v in normalized_flag:
                col_map[vcol] = normalized_flag[norm_v]
            else:
                print(f"[WARN] No matching FLAG column for {vcol}")
    
        # -------- 3) FLAG = 1 → NaN 처리 --------
        for vcol, fcol in col_map.items():
            mask = train_flags[fcol] == 1
            values_nan.loc[mask, vcol] = np.nan
    
        return values_nan
    # -------------------------------------------------------
    # 2. NaN 처리 (testset value만 보간)
    # -------------------------------------------------------
    def interpolate_test(self, df):
        df = df.sort_values(self.time_col).reset_index(drop=True)
        df = df.set_index(self.time_col)

        for col in df.columns:
            df[col] = df[col].interpolate()

        return df.reset_index()
    
    # -------------------------------------------------------
    # 3. DIR → U/V 변환 (utils)
    # -------------------------------------------------------
    def convert_dir_uv(self, df):
        return convert_dir_to_uv(df)

    def convert_flag_dir_to_uv(self, df):
        return convert_flag_dir_to_uv(df)
    
    # -------------------------------------------------------
    # 4. 정규화 (train fit → test transform)
    # -------------------------------------------------------
    def normalize(self, train_df, test_df):
        """MinMaxScaler 적용 (옵션 OFF 시 스킵)"""
        if not self.use_normalize:
            print("[INFO] Skipping normalization (use_normalize=False)")
            return train_df, test_df

        numeric_cols = [c for c in train_df.columns if c != self.time_col]

        self.scaler = MinMaxScaler()

        train_df[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols]  = self.scaler.transform(test_df[numeric_cols])

        # scaler 저장
        if self.save_dir is not None:
            scaler_path = os.path.join(self.save_dir, "scaler", "minmax_scaler.joblib")
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            print(f"[INFO] Scaler saved to: {scaler_path}")

        return train_df, test_df

    # -------------------------------------------------------
    # 5. 시퀀스 생성 (stride 지원)
    # -------------------------------------------------------
    def make_sequences(self, vals_df, flags_df=None, stride=1):
        vals_df = vals_df.sort_values(self.time_col).reset_index(drop=True)
        values = vals_df.drop(columns=[self.time_col]).values
    
        if flags_df is not None:
            flags_df = flags_df.sort_values(self.time_col).reset_index(drop=True)
            flags = flags_df.drop(columns=[self.time_col]).values
        else:
            flags = None
    
        seq_list = []
        flag_list = []
    
        # stride 적용: i를 0, stride, 2*stride ... 로 이동
        max_i = len(values) - self.seq_len + 1
        for i in range(0, max_i, stride):
            seq = values[i:i + self.seq_len]
    
            if flags is not None:
                fseq = flags[i:i + self.seq_len]
            else:
                fseq = None
    
            seq_list.append(seq)
            flag_list.append(fseq)
    
        seq_array = np.asarray(seq_list)
        flag_array = np.asarray(flag_list) if flags is not None else None
    
        return seq_array, flag_array

    # -------------------------------------------------------
    # 6. train 시퀀스 중 NaN 있는 것 제거
    # -------------------------------------------------------
    def drop_nan_sequences(self, seqs, flag_seqs=None):
        mask = ~np.isnan(seqs).any(axis=(1,2))
        seqs = seqs[mask]
        if flag_seqs is not None:
            flag_seqs = flag_seqs[mask]
        return seqs, flag_seqs

    # -------------------------------------------------------
    # 7. 시퀀스 shuffle + train/val split
    # -------------------------------------------------------
    def split_train_val(self, seqs, flag_seqs):
        X_train, X_val, F_train, F_val = train_test_split(
            seqs, flag_seqs,
            test_size=self.val_ratio,
            shuffle=True,
            random_state=self.random_state
        )
        return X_train, X_val, F_train, F_val
    
    def random_split_sequences(self, seqs, flags_tranad, val_ratio, seed):
        np.random.seed(seed)
        n = len(seqs)
    
        # 인덱스 섞기
        indices = np.random.permutation(n)
    
        # validation 개수
        n_val = int(n * val_ratio)
    
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
    
        # split
        return (
            seqs[train_idx],
            seqs[val_idx],
            flags_tranad[train_idx],
            flags_tranad[val_idx],
        )
    # -------------------------------------------------------
    # ★ TranAD용 데이터 처리
    # -------------------------------------------------------
    def build(self, train_vals, train_flags, test_vals, test_flags):
            
        # 1. DIR→UV
        train_vals = self.convert_dir_uv(train_vals)
        test_vals  = self.convert_dir_uv(test_vals)
        train_flags = self.convert_flag_dir_to_uv(train_flags)
        test_flags = self.convert_flag_dir_to_uv(test_flags)
        
        # 2. remove_outlier
        train_vals = self.apply_physical_limits(train_vals)
        
        # 3. Train: FLAG == 1 → Value를 np.nan으로 변환 (옵션)
        train_vals = train_vals.sort_values(self.time_col).reset_index(drop=True)
        train_flags = train_flags.sort_values(self.time_col).reset_index(drop=True)
    
        if self.apply_flag_nan:
        
            # 적용 전 상태 카운트
            prev_nan_count = train_vals.isna().sum().sum()
        
            # FLAG=1(오류) → NaN 적용
            train_vals = self.apply_flag_qc(train_vals, train_flags)
        
            # 적용 후 상태 카운트
            after_nan_count = train_vals.isna().sum().sum()

            print("\n===== FLAG 기반 오류 값 NaN 적용 =====")
            print(f"증가한 NaN 개수 : {after_nan_count - prev_nan_count}")

        # 4. testset Value는 항상 보간
        test_vals  = self.interpolate_test(test_vals)
    
        # 5. normalize + 스케일러 저장하기
        train_vals, test_vals = self.normalize(train_vals, test_vals)
    
        # 6. sequence 생성
        train_seq, train_flag_seq = self.make_sequences(train_vals, train_flags, stride=180)
        test_seq,  test_flag_seq  = self.make_sequences(test_vals,  test_flags, stride=1)
    
        # 7. NaN 시퀀스 제거
        before_drop = train_seq.shape[0]  
        train_seq, train_flag_seq = self.drop_nan_sequences(train_seq, train_flag_seq)
        after_drop = train_seq.shape[0]
        
        print(f"[INFO] Train Seq Before DropNaN: {before_drop}")
        print(f"[INFO] Train Seq After  DropNaN: {after_drop}")
        print(f"[INFO] Removed Sequences: {before_drop - after_drop}")
        
        # 8. train/val split (랜덤 split, 시퀀스 내부 순서는 그대로)
        seq_train, seq_val, flag_train_tranad, flag_val_tranad = \
            self.random_split_sequences(
                train_seq,
                train_flag_seq,
                self.val_ratio,
                self.random_state
            )
            
        return {
            "value_train": seq_train,
            "value_val":   seq_val,
            "value_test":  test_seq,
    
            "flag_train_tranad": flag_train_tranad,
            "flag_val_tranad":   flag_val_tranad,
            "flag_test_tranad":  test_flag_seq,
        }



import json

def save_tranad_model(model, optimizer, scheduler, save_dir, best_epoch, extra_info=None):
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"tranad_epoch{best_epoch}.pt")

    save_dict = {
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "extra": extra_info,
    }

    torch.save(save_dict, save_path)
    print(f"[INFO] Best TranAD Model saved → {save_path}")
    
def load_tranad_model(model, optimizer, scheduler, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", None)
    extra = checkpoint.get("extra", None)

    print(f"[INFO] TranAD Model Loaded from: {load_path}")
    return model, optimizer, scheduler, epoch, extra


def save_train_scores(train_scores_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "train_scores.json")

    save_obj = {str(k): v.tolist() for k, v in train_scores_dict.items()}

    with open(path, "w") as f:
        json.dump(save_obj, f)

    print(f"[INFO] train_scores saved → {path}")
    
def load_train_scores(save_dir):
    path = os.path.join(save_dir, "train_scores.json")

    if not os.path.exists(path):
        print("[WARN] train_scores.json not found!")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    # convert key to int, values to np.array
    train_scores = {int(k): np.array(v) for k, v in data.items()}

    print(f"[INFO] train_scores loaded → {path}")
    return train_scores

def save_test_scores(test_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "test_scores.npy")
    np.save(path, test_scores)
    print(f"[INFO] test_scores saved → {path}")

def load_test_scores(save_dir):
    path = os.path.join(save_dir, "test_scores.npy")
    if not os.path.exists(path):
        print("[WARN] test_scores.npy not found!")
        return None
    print(f"[INFO] test_scores loaded → {path}")
    return np.load(path)

def save_anomaly_pred(anomaly_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "anomaly_pred.npy")
    np.save(path, anomaly_pred)
    print(f"[INFO] anomaly_pred saved → {path}")

def load_anomaly_pred(save_dir):
    path = os.path.join(save_dir, "anomaly_pred.npy")
    if not os.path.exists(path):
        print("[WARN] anomaly_pred.npy not found!")
        return None
    print(f"[INFO] anomaly_pred loaded → {path}")
    return np.load(path)
#%% Setting
# 다변량
# data = 'Sea_level' #(Sea_level, SMB1_SMB2_AIR, SMB1_SMB2_Surface, SMB3_SMB4_WATER, SMB3_SMB4_Surface)
# train_data_dir = str(DATA_FOR_AI_DIR / "df" / f"{data}_train")
# test_data_dir = str(DATA_FOR_AI_DIR / "df" / f"{data}_test")

# 단변량
data = 'SMB1_AIR'
# SEA_LEVEL_LAKE, SEA_LEVEL_SEA
# SMB1_AIR, SMB2_AIR, SMB1_SURFACE, SMB2_SURFACE, 
# SMB3_WATER, SMB4_WATER
# SMB3_SURFACE, SMB4_SURFACE
train_data_dir = str(DATA_FOR_AI_DIR / "df_univariate" / f"{data}_train")
test_data_dir = str(DATA_FOR_AI_DIR / "df_univariate" / f"{data}_test")

train_pairs = list_pairs(train_data_dir)
print("사용 가능한 데이터:", train_pairs)
selected = input("사용할 데이터를 선택하세요 : ").strip()


#%%
# BiLSTM-AE 설정
seq_length = 8
learning_rate = 0.001
n_epochs = 200
batch_size = 32


now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# save_dir = str(OUTPUT_DIR / "bilstm" / f"usedb-2qc_seql{seq_length}_{data}_{selected}_{now_str}")
save_dir = str(OUTPUT_DIR / "bilstm" / "SMB1_AIR_solar" / "usedb-2qc_seql8_nlayer3_hsize128_SMB1_AIR_solar_20260312_152521")
best_epoch = 77

#%% 선택된 데이터 불러오기 (오래걸림)
train_values, train_flags, test_values, test_flags = load_selected_pair(train_data_dir, test_data_dir, selected)
test_1qc_flags_path = os.path.join(test_data_dir, f"{selected}_flags_1qc.csv")
def read_csv_safe(path):
    """cp949 → utf-8-sig 순으로 시도"""
    try:
        return pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
        
test_1qc_flags_df  = read_csv_safe(test_1qc_flags_path)


# 칼럼 순서 맞추기
cols = sorted(train_values.columns)
train_values = train_values[cols]
test_values  = test_values[cols]
cols_f = sorted(train_flags.columns)
train_flags = train_flags[cols_f]
test_flags  = test_flags[cols_f]

# 데이터 빌러 사용하여 AI용으로 정제
builder = AIDataBuilder(seq_len=seq_length, save_dir=save_dir, apply_flag_nan=True, use_normalize=True)

data = builder.build(
    train_values, train_flags,
    test_values,  test_flags
)

X_train = data["value_train"]
X_val   = data["value_val"]
X_test  = data["value_test"]

F_train_tranAD = data["flag_train_tranad"]
F_val_tranAD   = data["flag_val_tranad"]
F_test_tranAD  = data["flag_test_tranad"]


#%% 1. TranAD 학습
print("\n========== 학습 준비 ==========")
# 데이터 셋 준비 -------------------------------------------------
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(F_train_tranAD, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(F_val_tranAD, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(F_test_tranAD, dtype=torch.float32))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# -------------------------------------------------
# Model setup
n_features = X_train.shape[2]
seq_len    = builder.seq_len


from Bi_LSTM_auto_encoder import BiLSTMAutoEncoder

bilstm_ae_cpu = BiLSTMAutoEncoder(
    num_layers=3,       
    hidden_size=128,     
    nb_feature=n_features,
    dropout=0.2,          
    device=device).to(device)

optimizer = optim.AdamW(bilstm_ae_cpu.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

criterion = nn.MSELoss(reduction="none")
bilstm_ae = bilstm_ae_cpu.to(device)

#%% 모델 불러오기
load_path = f"{save_dir}/model/bilstm_ae_epoch{best_epoch}.pt"

checkpoint = torch.load(load_path, map_location=device)

bilstm_ae.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

last_epoch = checkpoint["epoch"]
best_val_loss = checkpoint.get("val_loss", None)

bilstm_ae = bilstm_ae.to(device).float()

# optimizer state도 float32로 변환 (중요)
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.float()
print(f"✅ BiLSTM-AE model loaded from epoch {last_epoch + 1}")
if best_val_loss is not None:
    print(f"   ↳ val_loss: {best_val_loss:.6f}")

#%%
# =========================
# 1) Anomaly score 계산
# =========================
def bilstm_ae_scores_featurewise(model, data_loader, device, criterion, mode="time_mean"):

    assert mode in ["time_mean", "last"]

    model.eval()
    scores_dict = {}

    with torch.no_grad():
        for batch_data, _ in tqdm(data_loader, desc=f"Collecting AE scores ({mode})"):
            batch_data = batch_data.to(device)   # (B, L, F)
            recon = model(batch_data)                     # (B, L, F)

            loss = criterion(recon, batch_data)           # (B, L, F)

            if mode == "time_mean":
                loss_feat = loss.mean(dim=1)              # (B, F)
            else:  # mode == "last"
                loss_feat = loss[:, -1, :]                # (B, F)

            loss_feat = loss_feat.detach().cpu().numpy()  # (B, F)

            F = loss_feat.shape[1]
            for f in range(F):
                if f not in scores_dict:
                    scores_dict[f] = []
                scores_dict[f].extend(loss_feat[:, f].tolist())

    for f in scores_dict:
        scores_dict[f] = np.asarray(scores_dict[f], dtype=float)

    return scores_dict

train_scores_np = bilstm_ae_scores_featurewise(
    bilstm_ae, train_loader, device, criterion, mode="time_mean"
)

test_scores_np = bilstm_ae_scores_featurewise(
    bilstm_ae, test_loader, device, criterion, mode="time_mean"
)

#%%
POT_CONFIG_PATH = os.path.join(BASE_DIR, "pot_configs.json")

def _pot_key(data_name, pair_name):
    return f"{data_name}/{pair_name}"

def _load_pot_scale(default_scale=1.5):
    if not os.path.exists(POT_CONFIG_PATH):
        return float(default_scale)
    try:
        with open(POT_CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        return float(cfg.get("pot_scaler", {}).get(_pot_key(data, selected), default_scale))
    except Exception:
        return float(default_scale)

pot_scaler = _load_pot_scale(default_scale=1.5)
print('scaler:', pot_scaler)

from pot import pot_eval, pot_eval_stable
# =========================
# 2) Threshold 계산
# =========================
def compute_threshold_percentile(train_scores_dict, q=97.5):

    threshold_dict = {}

    for f, scores in train_scores_dict.items():
        scores = np.asarray(scores, dtype=float)
        scores = scores[~np.isnan(scores)]  # 안전장치
        threshold_dict[f] = np.percentile(scores, q)

    return threshold_dict

def run_pot_on_scores(
    train_scores_dict,
    test_scores_dict,
    pot_func,
    q=1e-5,
    level=0.01):

    pot_results = {}
    anomaly_pred = {}

    for f, test_scores in test_scores_dict.items():
        train_scores = train_scores_dict[f]

        # 안전장치: numpy 변환
        train_scores = np.asarray(train_scores, dtype=float)
        test_scores = np.asarray(test_scores, dtype=float)

        # POT 구현상 label 필수 → dummy label
        dummy_label = np.zeros(len(test_scores), dtype=int)

        pot_results[f], anomaly_pred[f] = pot_func(
            init_score=train_scores,
            score=test_scores,
            label=dummy_label,
            q=q,
            level=level,
            th_scale=pot_scaler)
        

    return pot_results, anomaly_pred


precentile_thresholds = compute_threshold_percentile(train_scores_np, q=97.5)
pot_results, anomaly_pred = run_pot_on_scores(train_scores_dict=train_scores_np,test_scores_dict=test_scores_np, pot_func=pot_eval_stable)
pot_threshold = {
    f: result["threshold"]
    for f, result in pot_results.items()}


save_path = f"{save_dir}/csv/pot_info.txt"

with open(save_path, "w", encoding="utf-8") as f:
    f.write(f"pot_scaler: {pot_scaler}\n\n")
    f.write("pot_threshold:\n")
    
    for k, v in pot_threshold.items():
        f.write(f"feature_{k}: {v}\n")

print(f"Saved → {save_path}")
#%% 결과 저장코드 만들기
def make_ai_flag_final(test_scores, train_thresholds, test_values, selected, seq_len):
    """
    test_scores: {idx: np.array([...])}
    train_thresholds: {idx: th}
    test_values: test_values_df (날짜 포함 DF)
    selected: 사용자 선택 문자열 (ex: 'SMB1_SMB2_AIR_dir')
    seq_len: builder.seq_len
    """
    values_df = test_values.copy()
    # ================================================
    # 1) 방향데이터 여부 확인
    # ================================================
    if "dir" in selected.lower():
        test_values_uv = convert_dir_to_uv(test_values)
        date_col = None
        for c in test_values_uv.columns:
            if "날짜" in c or "date" in c.lower() or "time" in c.lower():
                date_col = c
                break
        
        # 날짜 제외한 나머지 컬럼만 feature로 사용
        feature_names = [c for c in test_values_uv.columns if c != date_col]

    else:
        feature_names = values_df.columns[:-1].tolist()


    # ================================================
    # 2) AI FLAG 생성 (기본: feature별 U,V 포함)
    # ================================================
    pred_dict = {}

    for f_idx, scores in test_scores.items():
        th = train_thresholds[f_idx]
        pred = np.where(scores > th, 4, 1)  # 4=이상, 1=정상
        pred_dict[f_idx] = pred

    test_AI_flag = pd.DataFrame(pred_dict)

    # 컬럼명 생성
    feature_names_ai = [f"{name}_AI_FLAG" for name in feature_names]

    # ================================================
    # 3) seq_len 패딩 수행
    # ================================================
    pad_len = seq_len - 1
    F = test_AI_flag.shape[1]
    pad = np.full((pad_len, F), np.nan)

    AI_flag_aligned = np.vstack([pad, test_AI_flag.values])
    test_AI_flag_aligned = pd.DataFrame(AI_flag_aligned, columns=feature_names_ai)

    # ================================================
    # 4) 방향 데이터인 경우 U/V FLAG 통합
    # ================================================
    if "dir" in selected.lower():

        merged_cols = {}
        final_cols = []

        for col in test_AI_flag_aligned.columns:

            # 예: SMB1_풍향_U_AI_FLAG → SMB1_풍향
            if "_U_AI_FLAG" in col or "_V_AI_FLAG" in col:
                base = col.split("_U_AI_FLAG")[0] if "_U_AI_FLAG" in col else col.split("_V_AI_FLAG")[0]

                if base not in merged_cols:
                    merged_cols[base] = []
                merged_cols[base].append(col)
            else:
                final_cols.append(col)

        # 통합 결과 저장 DF
        dir_flag_df = pd.DataFrame(index=test_AI_flag_aligned.index)

        for base, uv_cols in merged_cols.items():
            uv_vals = test_AI_flag_aligned[uv_cols].values
            u = uv_vals[:, 0]
            v = uv_vals[:, 1]
        
            # 둘 다 nan → nan
            both_nan = np.isnan(u) & np.isnan(v)
        
            # U/V 하나라도 4 → 4, 아니면 1
            merged_flag = np.where((u == 4) | (v == 4), 4, 1)
        
            # float으로 변경
            merged_flag = merged_flag.astype(float)
        
            # NaN 구간 처리
            merged_flag[both_nan] = np.nan
        
            dir_flag_df[f"{base}_AI_FLAG"] = merged_flag

        # 나머지 flag
        other_cols_df = test_AI_flag_aligned[final_cols]

        # 최종 병합
        test_AI_flag_aligned_final = pd.concat([other_cols_df, dir_flag_df], axis=1)

    else:
        test_AI_flag_aligned_final = test_AI_flag_aligned

    test_AI_QC_final = pd.concat(
        [test_values.reset_index(drop=True), test_AI_flag_aligned_final.reset_index(drop=True)],
        axis=1)
    # -----------------------------
    # "날짜" 컬럼을 항상 첫 번째로 이동
    # -----------------------------
    if "날짜" in test_AI_QC_final.columns:
        cols = ["날짜"] + [c for c in test_AI_QC_final.columns if c != "날짜"]
        test_AI_QC_final = test_AI_QC_final[cols]
    return test_AI_QC_final, test_AI_flag_aligned_final


# train threshold 저장
train_thresholds = pot_threshold

df_thr1 = pd.DataFrame(list(precentile_thresholds.items()), columns=["feature", "threshold"])   
df_thr1.to_csv(f"{save_dir}/csv/train_precentile_thresholds.csv", encoding='cp949', index=False)
df_thr2 = pd.DataFrame(list(pot_threshold.items()), columns=["feature", "threshold"])   
df_thr2.to_csv(f"{save_dir}/csv/train_pot_threshold.csv", encoding='cp949', index=False)

# AI QC 결과 저장
test_AI_QC_final_per, test_AI_flag_df = make_ai_flag_final(
    test_scores=test_scores_np,
    train_thresholds=precentile_thresholds,
    test_values=test_values,
    selected=selected,
    seq_len=seq_len
)
test_AI_QC_final_pot, test_AI_flag_df = make_ai_flag_final(
    test_scores=test_scores_np,
    train_thresholds=pot_threshold,
    test_values=test_values,
    selected=selected,
    seq_len=seq_len
)

test_AI_QC_final_per.to_csv(f"{save_dir}/csv/per_AI_Flag.csv", encoding='cp949', index=False)
test_AI_QC_final_pot.to_csv(f"{save_dir}/csv/pot_AI_Flag.csv", encoding='cp949', index=False)

#%% 정량적 평가 (정밀도, 재현율, F1-Score 계산)
# test_1qc_flags_df(1QC) VS test_flags(2QC)
df_1qc = test_1qc_flags_df.copy()
df_1qc["flag_1qc_bin"] = df_1qc[df_1qc.columns[1]].apply(
    lambda x: 4 if x >= 400 else 1
)

# test_AI_flag_df_per(AI QC) VS test_flags(2QC)
df_2qc = test_flags.copy()
df_2qc["flag_2qc_bin"] = df_2qc.iloc[:, 0].map({0: 1, 1: 4})

# test_AI_flag_df_pot(AI QC) VS test_flags(2QC)
df_aiqc_per = test_AI_QC_final_per.copy()
df_aiqc_pot = test_AI_QC_final_pot.copy()
df_aiqc_per = df_aiqc_per.rename(columns={df_aiqc_per.columns[-1]: "flag_ai_percentile"})
df_aiqc_pot = df_aiqc_pot.rename(columns={df_aiqc_pot.columns[-1]: "flag_ai_pot"})

# 합치기
df_merge = (
    df_2qc[["날짜", "flag_2qc_bin"]]
    .merge(df_1qc[["날짜", "flag_1qc_bin"]], on="날짜", how="inner")
    .merge(df_aiqc_per[["날짜", "flag_ai_percentile"]], on="날짜", how="inner")
    .merge(df_aiqc_pot[["날짜", "flag_ai_pot"]], on="날짜", how="inner")
    .merge(df_aiqc_pot[["날짜", df_aiqc_pot.columns[1]]], on="날짜", how="inner")
)
df_merge.to_csv(f'{save_dir}/csv/testset_total_flag.csv', index=False, encoding='cp949')

# 계산하기
from sklearn.metrics import precision_score, recall_score, f1_score

def calc_metrics(y_true, y_pred, pos_label=4):
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    return precision, recall, f1


def calc_all_metrics(df):
    # NaN 제거
    df = df[[
        "flag_2qc_bin",
        "flag_1qc_bin",
        "flag_ai_pot",
        "flag_ai_percentile"
    ]].dropna()

    y_true = df["flag_2qc_bin"].values

    # 1) 1QC
    y_pred_1qc = df["flag_1qc_bin"].values

    # 2) AI POT
    y_pred_ai_pot = df["flag_ai_pot"].values

    # 3) AI PER
    y_pred_ai_per = df["flag_ai_percentile"].values
    
    # 4) 1QC + AI(pot) (OR 조건)
    y_pred_combined = np.where(
        (y_pred_1qc == 4) | (y_pred_ai_pot == 4),
        4, 1
    )

    results = []

    for name, y_pred in [
        ("1QC", y_pred_1qc),
        ("AI_POT", y_pred_ai_pot),
        ("AI_PER", y_pred_ai_per),
        ("1QC_OR_AI", y_pred_combined),
    ]:
        precision, recall, f1 = calc_metrics(y_true, y_pred)

        results.append({
            "model": name,
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n_samples": len(y_true)
        })

    return pd.DataFrame(results)

metrics_total = calc_all_metrics(df_merge)
# metrics_total.to_csv(f'{save_dir}/csv/정량평가 결과2.csv', index=False, encoding='cp949')
print(metrics_total)
#%%
from fig_utils import plot_all_features_anomaly_with_flags, plot_all_features_anomaly_with_flags_custom, plot_ai_window

plot_all_features_anomaly_with_flags(
    test_with_AI_flag=test_AI_QC_final_per,
    test_flag_df=test_flags,
    test_1qc_flag_df=test_1qc_flags_df,
    save_dir=f"{save_dir}/fig_per"
)
plot_all_features_anomaly_with_flags(
    test_with_AI_flag=test_AI_QC_final_pot,
    test_flag_df=test_flags,
    test_1qc_flag_df=test_1qc_flags_df,
    save_dir=f"{save_dir}/fig_pot"
)

#%% 일정 부분만 확대해서 그리기
# plot_all_features_anomaly_with_flags_custom(
#     test_with_AI_flag=test_AI_QC_final_pot,
#     test_flag_df=test_flags,
#     test_1qc_flag_df=test_1qc_flags_df,
#     start_date='2025-05-07 10:00:00',
#     end_date='2025-05-07 12:00:00',
#     save_dir=f"{save_dir}/fig_pot"
# )
#%%
plot_ai_window(
    test_with_AI_flag=test_AI_QC_final_pot,
    test_flag_df=test_flags,
    test_1qc_flag_df=test_1qc_flags_df,
    start_date="2025-01-01",
    end_date="2025-12-31",
    save_dir=f"{save_dir}/fig_ai_windows_pot",
    pre_n=80,
    post_n=80,
    skip_no_ai=True
)
