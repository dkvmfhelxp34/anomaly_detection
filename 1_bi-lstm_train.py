# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 17:36:57 2025

@author: user
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
    data_dir   *_values.csv, *_flags.csv   pair  
    """
    files = os.listdir(data_dir)
    values = [f.replace("_values.csv", "") for f in files if f.endswith("_values.csv")]
    flags  = [f.replace("_flags.csv", "") for f in files if f.endswith("_flags.csv")]

    # values  flags    pair 
    pairs = sorted(list(set(values) & set(flags)))
    return pairs

def load_selected_pair(train_dir, test_dir, pair_name):
    """
    train_dir/test_dir  pair_name
    values/flags   4 DF .

    :
        train_values_df, train_flags_df,
        test_values_df,  test_flags_df
    """

    # -------------------------
    # Train  
    # -------------------------
    train_values_path = os.path.join(train_dir, f"{pair_name}_values.csv")
    train_flags_path  = os.path.join(train_dir, f"{pair_name}_flags.csv")

    # -------------------------
    # Test  
    # -------------------------
    test_values_path = os.path.join(test_dir, f"{pair_name}_values.csv")
    test_flags_path  = os.path.join(test_dir, f"{pair_name}_flags.csv")

    # -------------------------
    #    
    # -------------------------
    for p in [train_values_path, train_flags_path, test_values_path, test_flags_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f" : {p}")

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

    # Normalize first column name to a unified key.
    for df in [train_values_df, train_flags_df, test_values_df, test_flags_df]:
        if len(df.columns) > 0 and df.columns[0] != "date":
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

    #   ()
    if "" in train_values_df.columns:
        train_values_df = train_values_df.sort_values("date")
    if "" in train_flags_df.columns:
        train_flags_df = train_flags_df.sort_values("date")

    if "" in test_values_df.columns:
        test_values_df = test_values_df.sort_values("date")
    if "" in test_flags_df.columns:
        test_flags_df = test_flags_df.sort_values("date")

    return train_values_df, train_flags_df, test_values_df, test_flags_df


class AIDataBuilder:
    def __init__(self, seq_len=20, save_dir=None, time_col="date", val_ratio=0.2, random_state=42, apply_flag_nan=False, use_normalize=True):
        self.seq_len = seq_len
        self.time_col = time_col
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.apply_flag_nan = apply_flag_nan
        self.use_normalize = use_normalize
        self.scaler = None
        self.save_dir = save_dir
        
        #    
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
    
    # -------------------------------------------------------
    # 1. range QC 
    # -------------------------------------------------------
    def apply_physical_limits(self, df):
        return apply_physical_limits(df) 
    
    def apply_flag_qc(self, train_values, train_flags, time_col="date"):
        values_nan = train_values.copy()
    
        #   
        value_cols = [c for c in train_values.columns if c != time_col]
        flag_cols  = [c for c in train_flags.columns if c != time_col]
    
        # -------- 1) value   --------
        def normalize_value_name(name):
            # '_' 
            return name.replace("_", "")
    
        # -------- 2) flag   --------
        def normalize_flag_name(name):
            # FLAG   
            name = re.sub(r'FLAG', '', name)
            # '_' 
            name = name.replace("_", "")
            return name
    
        # FLAG     
        normalized_flag = {normalize_flag_name(fc): fc for fc in flag_cols}
    
        #  dictionary
        col_map = {}
    
        for vcol in value_cols:
            norm_v = normalize_value_name(vcol)
            if norm_v in normalized_flag:
                col_map[vcol] = normalized_flag[norm_v]
            else:
                print(f"[WARN] No matching FLAG column for {vcol}")
    
        # -------- 3) FLAG = 1  NaN  --------
        for vcol, fcol in col_map.items():
            mask = train_flags[fcol] == 1
            values_nan.loc[mask, vcol] = np.nan
    
        return values_nan
    # -------------------------------------------------------
    # 2. NaN  (testset value )
    # -------------------------------------------------------
    def interpolate_test(self, df):
        df = df.sort_values(self.time_col).reset_index(drop=True)
        df = df.set_index(self.time_col)

        for col in df.columns:
            df[col] = df[col].interpolate()

        return df.reset_index()
    
    # -------------------------------------------------------
    # 3. DIR  U/V  (utils)
    # -------------------------------------------------------
    def convert_dir_uv(self, df):
        return convert_dir_to_uv(df)

    def convert_flag_dir_to_uv(self, df):
        return convert_flag_dir_to_uv(df)
    
    # -------------------------------------------------------
    # 4.  (train fit  test transform)
    # -------------------------------------------------------
    def normalize(self, train_df, test_df):
        """MinMaxScaler  ( OFF  )"""
        if not self.use_normalize:
            print("[INFO] Skipping normalization (use_normalize=False)")
            return train_df, test_df

        numeric_cols = [c for c in train_df.columns if c != self.time_col]

        self.scaler = MinMaxScaler()

        train_df[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols]  = self.scaler.transform(test_df[numeric_cols])

        # scaler 
        if self.save_dir is not None:
            scaler_path = os.path.join(self.save_dir, "scaler", "minmax_scaler.joblib")
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            print(f"[INFO] Scaler saved to: {scaler_path}")

        return train_df, test_df

    # -------------------------------------------------------
    # 5.   (stride )
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
    
        # stride : i 0, stride, 2*stride ...  
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
    # 6. train   NaN   
    # -------------------------------------------------------
    def drop_nan_sequences(self, seqs, flag_seqs=None):
        mask = ~np.isnan(seqs).any(axis=(1,2))
        seqs = seqs[mask]
        if flag_seqs is not None:
            flag_seqs = flag_seqs[mask]
        return seqs, flag_seqs

    # -------------------------------------------------------
    # 7.  shuffle + train/val split
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
    
        #  
        indices = np.random.permutation(n)
    
        # validation 
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
    #  TranAD  
    # -------------------------------------------------------
    def build(self, train_vals, train_flags, test_vals, test_flags):
            
        # 1. DIRUV
        train_vals = self.convert_dir_uv(train_vals)
        test_vals  = self.convert_dir_uv(test_vals)
        train_flags = self.convert_flag_dir_to_uv(train_flags)
        test_flags = self.convert_flag_dir_to_uv(test_flags)
        
        # 2. remove_outlier
        train_vals = self.apply_physical_limits(train_vals)
        
        # 3. Train: FLAG == 1  Value np.nan  ()
        train_vals = train_vals.sort_values(self.time_col).reset_index(drop=True)
        train_flags = train_flags.sort_values(self.time_col).reset_index(drop=True)
    
        if self.apply_flag_nan:
        
            #    
            prev_nan_count = train_vals.isna().sum().sum()
        
            # FLAG=1()  NaN 
            train_vals = self.apply_flag_qc(train_vals, train_flags)
        
            #    
            after_nan_count = train_vals.isna().sum().sum()

            print("\n===== FLAG    NaN  =====")
            print(f" NaN  : {after_nan_count - prev_nan_count}")

        # 4. testset Value  
        test_vals  = self.interpolate_test(test_vals)
    
        # 5. normalize +  
        train_vals, test_vals = self.normalize(train_vals, test_vals)
    
        # 6. sequence 
        train_seq, train_flag_seq = self.make_sequences(train_vals, train_flags, stride=2)
        test_seq,  test_flag_seq  = self.make_sequences(test_vals,  test_flags, stride=1)
    
        # 7. NaN  
        before_drop = train_seq.shape[0]  
        train_seq, train_flag_seq = self.drop_nan_sequences(train_seq, train_flag_seq)
        after_drop = train_seq.shape[0]
        
        print(f"[INFO] Train Seq Before DropNaN: {before_drop}")
        print(f"[INFO] Train Seq After  DropNaN: {after_drop}")
        print(f"[INFO] Removed Sequences: {before_drop - after_drop}")
        
        # 8. train/val split ( split,    )
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
    print(f"[INFO] Best TranAD Model saved  {save_path}")
    
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

    print(f"[INFO] train_scores saved  {path}")
    
def load_train_scores(save_dir):
    path = os.path.join(save_dir, "train_scores.json")

    if not os.path.exists(path):
        print("[WARN] train_scores.json not found!")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    # convert key to int, values to np.array
    train_scores = {int(k): np.array(v) for k, v in data.items()}

    print(f"[INFO] train_scores loaded  {path}")
    return train_scores

def save_test_scores(test_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "test_scores.npy")
    np.save(path, test_scores)
    print(f"[INFO] test_scores saved  {path}")

def load_test_scores(save_dir):
    path = os.path.join(save_dir, "test_scores.npy")
    if not os.path.exists(path):
        print("[WARN] test_scores.npy not found!")
        return None
    print(f"[INFO] test_scores loaded  {path}")
    return np.load(path)

def save_anomaly_pred(anomaly_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "anomaly_pred.npy")
    np.save(path, anomaly_pred)
    print(f"[INFO] anomaly_pred saved  {path}")

def load_anomaly_pred(save_dir):
    path = os.path.join(save_dir, "anomaly_pred.npy")
    if not os.path.exists(path):
        print("[WARN] anomaly_pred.npy not found!")
        return None
    print(f"[INFO] anomaly_pred loaded  {path}")
    return np.load(path)
#%% Setting
# 
# data = 'Sea_level' #(Sea_level, SMB1_SMB2_AIR, SMB1_SMB2_Surface, SMB3_SMB4_WATER, SMB3_SMB4_Surface)
# train_data_dir = str(DATA_FOR_AI_DIR / "df" / f"{data}_train")
# test_data_dir = str(DATA_FOR_AI_DIR / "df" / f"{data}_test")

# 
data = 'SMB1_AIR'
# SEA_LEVEL_LAKE, SEA_LEVEL_SEA
# SMB1_AIR, SMB2_AIR, SMB1_SURFACE, SMB2_SURFACE, 
# SMB3_WATER, SMB4_WATER
# SMB3_SURFACE, SMB4_SURFACE
train_data_dir = str(DATA_FOR_AI_DIR / "df_univariate" / f"{data}_train")
test_data_dir = str(DATA_FOR_AI_DIR / "df_univariate" / f"{data}_test")

train_pairs = list_pairs(train_data_dir)
print("  :", train_pairs)
selected = input("   : ").strip()


#%%
# BiLSTM-AE 
seq_length = 8
learning_rate = 0.001
n_epochs = 200
batch_size = 32
h_size = 128
n_layers = 3

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = str(OUTPUT_DIR / "bilstm" / f"{data}_{selected}" / f"usedb-2qc_seql{seq_length}_nlayer{n_layers}_hsize{h_size}_{data}_{selected}_{now_str}")
subfolders = ["fig_per", "fig_pot", "csv", "model", "scaler"]

#  
for folder in subfolders:
    path = os.path.join(save_dir, folder)
    os.makedirs(path, exist_ok=True)


#%%    ()
train_values, train_flags, test_values, test_flags = load_selected_pair(train_data_dir, test_data_dir, selected)
test_1qc_flags_path = os.path.join(test_data_dir, f"{selected}_flags_1qc.csv")
def read_csv_safe(path):
    """cp949  utf-8-sig  """
    try:
        return pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
        
test_1qc_flags_df  = read_csv_safe(test_1qc_flags_path)


#   
cols = sorted(train_values.columns)
train_values = train_values[cols]
test_values  = test_values[cols]
cols_f = sorted(train_flags.columns)
train_flags = train_flags[cols_f]
test_flags  = test_flags[cols_f]

#    AI 
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

#%%
# import numpy as np
# import matplotlib.pyplot as plt

# values_clean = train_values.copy()
# mask = train_flags["SMB3_FLAG_chlorophyll"] == 1
# values_clean.loc[mask, "SMB3_chlorophyll"] = np.nan

# data = values_clean["SMB3_chlorophyll"].dropna()

# bins = np.arange(0, data.max() + 10, 10)

# plt.figure(figsize=(12,4))

# # % 
# weights = np.ones_like(data) * 100 / len(data)
# plt.hist(
#     data,
#     bins=bins,
#     weights=weights,
#     edgecolor="black",   #  
#     linewidth=0.8        #  
# )
# plt.xlim(0,440)

# # X 10 
# plt.xticks(np.arange(0, data.max() + 10, 10), rotation=30)
# plt.xlabel("SMB4_chlorophyll")
# plt.ylabel("Percentage (%)")
# plt.title("Distribution of SMB3_chlorophyll (Normal data only)")

# plt.show()

# percent_under_50 = (data <= 50).sum() / len(data) * 100
# print(f"50   : {percent_under_50:.2f}%")

#%% 1. TranAD 
print("\n==========   ==========")
#    -------------------------------------------------
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
    num_layers=n_layers,       
    hidden_size=h_size,     
    nb_feature=n_features,
    dropout=0.2,          
    device=device).to(device)

optimizer = optim.AdamW(bilstm_ae_cpu.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

criterion = nn.MSELoss(reduction="none")
bilstm_ae = bilstm_ae_cpu.to(device)

# %%
# ======================
# Early Stopping 
# ======================
patience = 15       
wait = 0               #    epoch 
early_stop = False

best_val_loss = float("inf")
best_epoch = -1
best_state = None

# ======================
# Training Loop
# ======================
for epoch in range(n_epochs):
    bilstm_ae.train()
    epoch_losses = []

    for batch_data, _ in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{n_epochs}",
        leave=True
    ):
        batch_data = batch_data.to(device)

        recon = bilstm_ae(batch_data)

        loss_batch = criterion(recon, batch_data).mean(dim=(1, 2))  # (B,)
        loss = loss_batch.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    scheduler.step()
    mean_train_loss = np.mean(epoch_losses)
    tqdm.write(f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.6f}")

    # ---------------------
    # Validation
    # ---------------------
    bilstm_ae.eval()
    val_losses = []

    with torch.no_grad():
        for batch_data, _ in val_loader:
            batch_data = batch_data.to(device)
            recon = bilstm_ae(batch_data)

            val_loss_batch = criterion(recon, batch_data).mean(dim=(1, 2))  # (B,)
            val_losses.extend(val_loss_batch.detach().cpu().numpy())

    mean_val_loss = np.mean(val_losses)
    tqdm.write(f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.6f}")

    # ---------------------
    # Best model & Early Stopping
    # ---------------------
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        best_epoch = epoch
        wait = 0  #  

        best_state = {
            "epoch": epoch,
            "model_state_dict": bilstm_ae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        tqdm.write(
            f"   Best model updated at epoch {epoch+1} "
            f"(val_loss={best_val_loss:.6f})"
        )
    else:
        wait += 1
        tqdm.write(f"   No improvement ({wait}/{patience})")

        if wait >= patience:
            tqdm.write(
                f"\n Early stopping triggered at epoch {epoch+1} "
                f"(best_epoch={best_epoch+1})"
            )
            early_stop = True

    if early_stop:
        break

# %%   
print("\n Final Best Epoch:", best_epoch + 1)
print(" Training Finished!")

#  
save_path = f"{save_dir}/model/bilstm_ae_epoch{best_epoch+1}.pt"

torch.save(
    {
        "epoch": best_epoch,
        "model_state_dict": best_state["model_state_dict"],
        "optimizer_state_dict": best_state["optimizer_state_dict"],
        "scheduler_state_dict": best_state["scheduler_state_dict"],
        "val_loss": best_val_loss
    },
    save_path
)

print(f"\n Best BiLSTM-AE model saved to: {save_path}")

#%%  
load_path = f"{save_dir}/model/bilstm_ae_epoch{best_epoch+1}.pt"

checkpoint = torch.load(load_path, map_location=device)

bilstm_ae.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

last_epoch = checkpoint["epoch"]
best_val_loss = checkpoint.get("val_loss", None)

bilstm_ae = bilstm_ae.to(device).float()

# optimizer state float32  ()
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.float()
            
print(f" BiLSTM-AE model loaded from epoch {last_epoch + 1}")
if best_val_loss is not None:
    print(f"    val_loss: {best_val_loss:.6f}")

#%%
# =========================
# 1) Anomaly score 
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
    bilstm_ae, train_loader, device, criterion, mode="last"
)

test_scores_np = bilstm_ae_scores_featurewise(
    bilstm_ae, test_loader, device, criterion, mode="last"
)

#%%
pot_scaler=1
print('scaler:', pot_scaler)

from pot import pot_eval, pot_eval_stable
# =========================
# 2) Threshold 
# =========================
def compute_threshold_percentile(train_scores_dict, q=97.5):

    threshold_dict = {}

    for f, scores in train_scores_dict.items():
        scores = np.asarray(scores, dtype=float)
        scores = scores[~np.isnan(scores)]  # 
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

        # : numpy 
        train_scores = np.asarray(train_scores, dtype=float)
        test_scores = np.asarray(test_scores, dtype=float)

        # POT  label   dummy label
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

print(f"Saved  {save_path}")
#%%   
def make_ai_flag_final(test_scores, train_thresholds, test_values, selected, seq_len):
    """
    test_scores: {idx: np.array([...])}
    train_thresholds: {idx: th}
    test_values: test_values_df (  DF)
    selected:    (ex: 'SMB1_SMB2_AIR_dir')
    seq_len: builder.seq_len
    """
    values_df = test_values.copy()
    # ================================================
    # 1)   
    # ================================================
    if "dir" in selected.lower():
        test_values_uv = convert_dir_to_uv(test_values)
        date_col = None
        for c in test_values_uv.columns:
            if "date" in c.lower() or "time" in c.lower():
                date_col = c
                break
        
        #     feature 
        feature_names = [c for c in test_values_uv.columns if c != date_col]

    else:
        feature_names = values_df.columns[:-1].tolist()


    # ================================================
    # 2) AI FLAG  (: feature U,V )
    # ================================================
    pred_dict = {}

    for f_idx, scores in test_scores.items():
        th = train_thresholds[f_idx]
        pred = np.where(scores > th, 4, 1)  # 4=, 1=
        pred_dict[f_idx] = pred

    test_AI_flag = pd.DataFrame(pred_dict)

    #  
    feature_names_ai = [f"{name}_AI_FLAG" for name in feature_names]

    # ================================================
    # 3) seq_len  
    # ================================================
    pad_len = seq_len - 1
    F = test_AI_flag.shape[1]
    pad = np.full((pad_len, F), np.nan)

    AI_flag_aligned = np.vstack([pad, test_AI_flag.values])
    test_AI_flag_aligned = pd.DataFrame(AI_flag_aligned, columns=feature_names_ai)

    # ================================================
    # 4)    U/V FLAG 
    # ================================================
    if "dir" in selected.lower():

        merged_cols = {}
        final_cols = []

        for col in test_AI_flag_aligned.columns:

            # : SMB1__U_AI_FLAG  SMB1_
            if "_U_AI_FLAG" in col or "_V_AI_FLAG" in col:
                base = col.split("_U_AI_FLAG")[0] if "_U_AI_FLAG" in col else col.split("_V_AI_FLAG")[0]

                if base not in merged_cols:
                    merged_cols[base] = []
                merged_cols[base].append(col)
            else:
                final_cols.append(col)

        #    DF
        dir_flag_df = pd.DataFrame(index=test_AI_flag_aligned.index)

        for base, uv_cols in merged_cols.items():
            uv_vals = test_AI_flag_aligned[uv_cols].values
            u = uv_vals[:, 0]
            v = uv_vals[:, 1]
        
            #   nan  nan
            both_nan = np.isnan(u) & np.isnan(v)
        
            # U/V  4  4,  1
            merged_flag = np.where((u == 4) | (v == 4), 4, 1)
        
            # float 
            merged_flag = merged_flag.astype(float)
        
            # NaN  
            merged_flag[both_nan] = np.nan
        
            dir_flag_df[f"{base}_AI_FLAG"] = merged_flag

        #  flag
        other_cols_df = test_AI_flag_aligned[final_cols]

        #  
        test_AI_flag_aligned_final = pd.concat([other_cols_df, dir_flag_df], axis=1)

    else:
        test_AI_flag_aligned_final = test_AI_flag_aligned

    test_AI_QC_final = pd.concat(
        [test_values.reset_index(drop=True), test_AI_flag_aligned_final.reset_index(drop=True)],
        axis=1)
    # -----------------------------
    # ""     
    # -----------------------------
    if "date" in test_AI_QC_final.columns:
        cols = ["date"] + [c for c in test_AI_QC_final.columns if c != "date"]
        test_AI_QC_final = test_AI_QC_final[cols]
    return test_AI_QC_final, test_AI_flag_aligned_final


# train threshold 

df_thr1 = pd.DataFrame(list(precentile_thresholds.items()), columns=["feature", "threshold"])   
df_thr1.to_csv(f"{save_dir}/csv/train_precentile_thresholds.csv", encoding='cp949', index=False)
df_thr2 = pd.DataFrame(list(pot_threshold.items()), columns=["feature", "threshold"])   
df_thr2.to_csv(f"{save_dir}/csv/train_pot_threshold.csv", encoding='cp949', index=False)

# AI QC  
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

#%%   (, , F1-Score )
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

# 
df_merge = (
    df_2qc[["date", "flag_2qc_bin"]]
    .merge(df_1qc[["date", "flag_1qc_bin"]], on="date", how="inner")
    .merge(df_aiqc_per[["date", "flag_ai_percentile"]], on="date", how="inner")
    .merge(df_aiqc_pot[["date", "flag_ai_pot"]], on="date", how="inner")
    .merge(df_aiqc_pot[["date", df_aiqc_pot.columns[1]]], on="date", how="inner")
)
df_merge.to_csv(f'{save_dir}/csv/testset_total_flag.csv', index=False, encoding='cp949')

# 
from sklearn.metrics import precision_score, recall_score, f1_score

def calc_metrics(y_true, y_pred, pos_label=4):
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    return precision, recall, f1


def calc_all_metrics(df):
    # NaN 
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
    
    # 4) 1QC + AI(pot) (OR )
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
metrics_total.to_csv(f'{save_dir}/csv/ .csv', index=False, encoding='cp949')
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

#%%    
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

