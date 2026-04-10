from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SCR_DIR = ROOT / "scr"
if str(SCR_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCR_DIR))

from pot import pot_eval_stable  # type: ignore


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv_safe(path: Path) -> pd.DataFrame:
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path)


def detect_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc or c == "날짜":
            return c
    return df.columns[0]


def list_pairs(data_dir: Path) -> List[str]:
    files = [p.name for p in data_dir.glob("*.csv")]
    value_names = {f[: -len("_values.csv")] for f in files if f.endswith("_values.csv")}
    flag_names = {f[: -len("_flags.csv")] for f in files if f.endswith("_flags.csv")}
    return sorted(value_names & flag_names)


def load_pair(train_dir: Path, test_dir: Path, pair: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_values = read_csv_safe(train_dir / f"{pair}_values.csv")
    train_flags = read_csv_safe(train_dir / f"{pair}_flags.csv")
    test_values = read_csv_safe(test_dir / f"{pair}_values.csv")
    test_flags = read_csv_safe(test_dir / f"{pair}_flags.csv")
    return train_values, train_flags, test_values, test_flags


def _normalize_name(name: str, drop_flag: bool = False) -> str:
    x = name.lower().replace("_", "")
    if drop_flag:
        x = x.replace("flag", "")
    return x


def apply_flag_nan(values: pd.DataFrame, flags: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = values.copy()
    value_cols = [c for c in values.columns if c != date_col]
    flag_cols = [c for c in flags.columns if c != date_col]
    rev = {_normalize_name(c, drop_flag=True): c for c in flag_cols}
    for vcol in value_cols:
        key = _normalize_name(vcol)
        if key in rev:
            mask = pd.to_numeric(flags[rev[key]], errors="coerce").fillna(0).astype(float) == 1
            out.loc[mask, vcol] = np.nan
    return out


def preprocess_values(
    train_values: pd.DataFrame,
    train_flags: pd.DataFrame,
    test_values: pd.DataFrame,
    date_col: str,
    apply_flag_mask: bool,
    use_normalize: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    train = train_values.copy()
    test = test_values.copy()

    if apply_flag_mask:
        train = apply_flag_nan(train, train_flags, date_col)

    train = train.sort_values(date_col).reset_index(drop=True)
    test = test.sort_values(date_col).reset_index(drop=True)

    for col in [c for c in test.columns if c != date_col]:
        test[col] = pd.to_numeric(test[col], errors="coerce").interpolate(limit_direction="both")

    for col in [c for c in train.columns if c != date_col]:
        train[col] = pd.to_numeric(train[col], errors="coerce").interpolate(limit_direction="both")

    scaler = None
    if use_normalize:
        scaler = MinMaxScaler()
        cols = [c for c in train.columns if c != date_col]
        train[cols] = scaler.fit_transform(train[cols])
        test[cols] = scaler.transform(test[cols])

    return train, test, scaler


def to_sequences(values_df: pd.DataFrame, flags_df: pd.DataFrame, seq_len: int, date_col: str) -> Tuple[np.ndarray, np.ndarray]:
    x = values_df.drop(columns=[date_col]).to_numpy(dtype=np.float32)
    fdf = flags_df.copy()
    if date_col in fdf.columns:
        fdf = fdf.drop(columns=[date_col])
    f = fdf.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    if x.shape[0] < seq_len:
        raise ValueError(f"Rows({x.shape[0]}) must be >= seq_len({seq_len}).")

    xs = []
    ys = []
    for i in range(seq_len - 1, len(x)):
        st = i - seq_len + 1
        xs.append(x[st : i + 1])
        ys.append(float(np.nanmax(f[st : i + 1])) if f.size > 0 else 0.0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class BiLSTMAutoEncoder(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, nb_feature: int, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=nb_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.h_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.c_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=nb_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.out = nn.Linear(hidden_size, nb_feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        h0 = torch.zeros((self.num_layers * 2, b, self.hidden_size), device=x.device)
        c0 = torch.zeros((self.num_layers * 2, b, self.hidden_size), device=x.device)
        _, (h, c) = self.encoder(x, (h0, c0))

        h = h.view(self.num_layers, 2, b, self.hidden_size)
        c = c.view(self.num_layers, 2, b, self.hidden_size)
        h = self.h_proj(torch.cat([h[:, 0], h[:, 1]], dim=-1))
        c = self.c_proj(torch.cat([c[:, 0], c[:, 1]], dim=-1))

        out = torch.zeros_like(x)
        dec_in = x[:, -1:, :]
        hidden = (h, c)
        for i in range(l - 1, -1, -1):
            dec_out, hidden = self.decoder(dec_in, hidden)
            dec_out = self.out(dec_out)
            out[:, i : i + 1, :] = dec_out
            dec_in = dec_out
        return out


@dataclass
class RunConfig:
    data_name: str
    pair: str
    data_root: str
    output_root: str
    seq_len: int = 8
    batch_size: int = 32
    n_layers: int = 3
    hidden_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 50
    val_ratio: float = 0.2
    patience: int = 10
    pot_scale: float = 1.0
    pot_q: float = 1e-5
    pot_level: float = 0.01
    random_seed: int = 42
    use_flag_nan: bool = True
    use_normalize: bool = True


def _make_loaders(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, batch: int):
    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True),
        DataLoader(val_ds, batch_size=batch, shuffle=False),
        DataLoader(test_ds, batch_size=batch, shuffle=False),
    )


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str, epochs: int, lr: float, patience: int):
    criterion = nn.MSELoss(reduction="none")
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)

    best_loss = float("inf")
    best_state = None
    wait = 0

    model.to(device)
    for ep in range(epochs):
        model.train()
        for xb, _ in train_loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = criterion(recon, xb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                recon = model(xb)
                vals.append(criterion(recon, xb).mean().item())

        mean_val = float(np.mean(vals))
        scheduler.step()
        if mean_val < best_loss:
            best_loss = mean_val
            wait = 0
            best_state = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_loss,
                "epoch": ep,
            }
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed: no checkpoint state captured.")
    return best_state


def score_featurewise(model: nn.Module, loader: DataLoader, device: str, mode: str = "last") -> Dict[int, np.ndarray]:
    if mode not in {"last", "time_mean"}:
        raise ValueError("mode must be 'last' or 'time_mean'")
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    scores: Dict[int, List[float]] = {}
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = criterion(recon, xb)
            feat = loss[:, -1, :] if mode == "last" else loss.mean(dim=1)
            arr = feat.detach().cpu().numpy()
            for f in range(arr.shape[1]):
                scores.setdefault(f, []).extend(arr[:, f].tolist())
    return {k: np.asarray(v, dtype=float) for k, v in scores.items()}


def compute_percentile_thresholds(train_scores: Dict[int, np.ndarray], q: float = 97.5) -> Dict[int, float]:
    out = {}
    for f, s in train_scores.items():
        s = np.asarray(s, dtype=float)
        s = s[np.isfinite(s)]
        out[f] = float(np.percentile(s, q))
    return out


def compute_pot_thresholds(train_scores: Dict[int, np.ndarray], test_scores: Dict[int, np.ndarray], q: float, level: float, th_scale: float) -> Dict[int, float]:
    out = {}
    for f, s_test in test_scores.items():
        s_train = np.asarray(train_scores[f], dtype=float)
        s_test = np.asarray(s_test, dtype=float)
        dummy = np.zeros(len(s_test), dtype=int)
        result, _ = pot_eval_stable(
            init_score=s_train,
            score=s_test,
            label=dummy,
            q=q,
            level=level,
            th_scale=th_scale,
        )
        out[f] = float(result["threshold"])
    return out


def build_ai_flags(test_values: pd.DataFrame, test_scores: Dict[int, np.ndarray], thresholds: Dict[int, float], seq_len: int, date_col: str) -> pd.DataFrame:
    feature_cols = [c for c in test_values.columns if c != date_col]
    pred = {}
    for i, col in enumerate(feature_cols):
        s = np.asarray(test_scores[i], dtype=float)
        th = thresholds[i]
        p = np.where(s > th, 4.0, 1.0)
        pred[f"{col}_AI_FLAG"] = p

    flag_df = pd.DataFrame(pred)
    pad = pd.DataFrame(np.nan, index=range(seq_len - 1), columns=flag_df.columns)
    aligned = pd.concat([pad, flag_df], axis=0, ignore_index=True)
    merged = pd.concat([test_values.reset_index(drop=True), aligned], axis=1)
    ai_cols = [c for c in merged.columns if c.endswith("_AI_FLAG")]
    merged["AI_TOTAL_FLAG"] = np.where((merged[ai_cols] == 4).any(axis=1), 4.0, 1.0)
    merged.loc[merged[ai_cols].isna().all(axis=1), "AI_TOTAL_FLAG"] = np.nan
    return merged


def calc_metrics_table(df_flags_2qc: pd.DataFrame, df_ai: pd.DataFrame, date_col: str, df_flags_1qc: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    f2 = df_flags_2qc.copy()
    if date_col not in f2.columns:
        date_col = detect_date_col(f2)

    target_col_2qc = [c for c in f2.columns if c != date_col][0]
    f2 = f2[[date_col, target_col_2qc]].copy()
    f2["flag_2qc_bin"] = pd.to_numeric(f2[target_col_2qc], errors="coerce").map({0: 1, 1: 4})

    merged = f2[[date_col, "flag_2qc_bin"]].merge(df_ai[[date_col, "AI_TOTAL_FLAG"]], on=date_col, how="inner")

    rows = []
    y_true = merged["flag_2qc_bin"].to_numpy()
    y_ai = merged["AI_TOTAL_FLAG"].to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_ai)
    y_true = y_true[mask]
    y_ai = y_ai[mask]

    rows.append(
        {
            "model": "AI_TOTAL",
            "precision": precision_score(y_true, y_ai, pos_label=4, zero_division=0),
            "recall": recall_score(y_true, y_ai, pos_label=4, zero_division=0),
            "f1": f1_score(y_true, y_ai, pos_label=4, zero_division=0),
            "n_samples": int(len(y_true)),
        }
    )

    if df_flags_1qc is not None and date_col in df_flags_1qc.columns:
        f1q = df_flags_1qc.copy()
        c1 = [c for c in f1q.columns if c != date_col][0]
        f1q = f1q[[date_col, c1]].copy()
        f1q["flag_1qc_bin"] = np.where(pd.to_numeric(f1q[c1], errors="coerce") >= 400, 4, 1)
        merged2 = merged.merge(f1q[[date_col, "flag_1qc_bin"]], on=date_col, how="inner")
        y1 = merged2["flag_1qc_bin"].to_numpy()
        yt = merged2["flag_2qc_bin"].to_numpy()
        ma = np.isfinite(y1) & np.isfinite(yt)
        rows.append(
            {
                "model": "1QC",
                "precision": precision_score(yt[ma], y1[ma], pos_label=4, zero_division=0),
                "recall": recall_score(yt[ma], y1[ma], pos_label=4, zero_division=0),
                "f1": f1_score(yt[ma], y1[ma], pos_label=4, zero_division=0),
                "n_samples": int(ma.sum()),
            }
        )

    return pd.DataFrame(rows)


def default_run_dir(output_root: Path, data_name: str, pair: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{data_name}_{pair}_{ts}"


def save_run_config(path: Path, cfg: RunConfig) -> None:
    path.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")


def load_run_config(path: Path) -> RunConfig:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return RunConfig(**obj)


def ensure_dirs(run_dir: Path) -> None:
    for p in [run_dir, run_dir / "model", run_dir / "csv", run_dir / "scaler"]:
        p.mkdir(parents=True, exist_ok=True)


def save_thresholds(path: Path, thresholds: Dict[int, float]) -> None:
    pd.DataFrame({"feature": list(thresholds.keys()), "threshold": list(thresholds.values())}).to_csv(path, index=False, encoding="utf-8-sig")
