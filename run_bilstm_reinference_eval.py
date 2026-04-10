import argparse
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt

from Bi_LSTM_auto_encoder import BiLSTMAutoEncoder
from fig_utils import plot_all_features_anomaly_with_flags
from pot import pot_eval_stable
from range_outlier_remove import apply_physical_limits


DATE_COL_ALIAS = "date"
PLOT_DATE_COL = "날짜"
PSEUDO_START = pd.Timestamp("2024-01-01 00:00:00")
PSEUDO_END = pd.Timestamp("2024-12-31 23:59:59")
PSEUDO_TARGETS = {"temp", "Salinity", "pH", "O2ppm", "O2Per", "chl-a"}


@dataclass
class ExperimentItem:
    top_folder: str
    experiment_folder: str
    experiment_path: Path
    model_path: Path
    scaler_path: Optional[Path]
    data_name: str
    pair: str
    pot_key: str
    pot_scaler: float
    seq_len: int


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Failed to read CSV with known encodings: {path}")


def detect_date_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in cols:
        s = str(c).lower()
        if "날짜" in str(c) or "date" in s or "time" in s or "obs_time" in s:
            return c
    return cols[0]


def normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dcol = detect_date_col(df)
    if dcol != DATE_COL_ALIAS:
        df = df.rename(columns={dcol: DATE_COL_ALIAS})
    df[DATE_COL_ALIAS] = pd.to_datetime(df[DATE_COL_ALIAS], errors="coerce")
    return df


def parse_top_folder(top_folder: str) -> Tuple[str, str]:
    if top_folder == "Sea_level_Repr_Sea":
        return "SEA_LEVEL_SEA", "Sea_level_Repr_Sea"
    if top_folder == "Sea_level_Repr_Lake":
        return "SEA_LEVEL_LAKE", "Sea_level_Repr_Lake"

    parts = top_folder.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse top folder: {top_folder}")
    data_name = f"{parts[0]}_{parts[1]}"
    pair = parts[2]
    return data_name, pair


def select_experiment_dir(top_dir: Path) -> Path:
    subdirs = [p for p in top_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No experiment subfolder in {top_dir}")
    if len(subdirs) == 1:
        return subdirs[0]

    o_dirs = [p for p in subdirs if p.name.startswith("O")]
    if o_dirs:
        return max(o_dirs, key=lambda p: p.stat().st_mtime)
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def select_model_file(ai_model_dir: Path) -> Path:
    cands = list(ai_model_dir.rglob("*.pt"))
    if not cands:
        raise FileNotFoundError(f"No .pt model in {ai_model_dir}")

    def epoch_of(path: Path) -> int:
        m = re.search(r"epoch(\d+)", path.name, flags=re.IGNORECASE)
        return int(m.group(1)) if m else -1

    return max(cands, key=lambda p: (epoch_of(p), p.stat().st_mtime))


def parse_seq_len(experiment_folder_name: str, default_seq: int = 8) -> int:
    m = re.search(r"seql(\d+)", experiment_folder_name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return default_seq


def infer_arch_from_ckpt(model_path: Path) -> Tuple[int, int]:
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    k0 = "encoder.lstm.weight_ih_l0"
    if k0 not in sd:
        raise KeyError(f"Missing key {k0} in {model_path}")
    hidden_size = int(sd[k0].shape[0] // 4)
    n_layers = len([k for k in sd.keys() if re.fullmatch(r"encoder\.lstm\.weight_ih_l\d+", k)])
    return n_layers, hidden_size


def load_pot_scales(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    return {k: float(v) for k, v in cfg.get("pot_scaler", {}).items()}


def load_split_frames(df_root: Path, data_name: str, pair: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = df_root / f"{data_name}_{split}"
    v = read_csv_auto(split_dir / f"{pair}_values.csv")
    f2 = read_csv_auto(split_dir / f"{pair}_flags.csv")
    f1 = read_csv_auto(split_dir / f"{pair}_flags_1qc.csv")
    return normalize_date_col(v), normalize_date_col(f2), normalize_date_col(f1)


def interpolate_series(s: pd.Series) -> pd.Series:
    return s.interpolate(limit_direction="both")


def build_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = len(arr)
    if n < seq_len:
        return np.empty((0, seq_len, arr.shape[1]), dtype=np.float32)
    seqs = [arr[i : i + seq_len] for i in range(0, n - seq_len + 1)]
    seqs = np.asarray(seqs, dtype=np.float32).reshape(-1, seq_len, arr.shape[1])
    return seqs


def score_sequences(model: torch.nn.Module, seqs: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    if len(seqs) == 0:
        return np.empty((0, 1), dtype=float)
    model.eval()
    criterion = torch.nn.MSELoss(reduction="none")
    out_scores = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = torch.from_numpy(seqs[i : i + batch_size]).to(device=device, dtype=torch.float32)
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=1)
            out_scores.append(loss.detach().cpu().numpy())
    return np.concatenate(out_scores, axis=0).astype(float)


def align_scores(scores: np.ndarray, n_rows: int, seq_len: int) -> np.ndarray:
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    aligned = np.full((n_rows, scores.shape[1]), np.nan, dtype=float)
    if len(scores) == 0:
        return aligned
    start = seq_len - 1
    end = min(n_rows, start + len(scores))
    aligned[start:end, :] = scores[: end - start, :]
    return aligned


def build_ai_flag(scores_aligned: np.ndarray, thresholds: np.ndarray, original_value: pd.Series) -> pd.Series:
    if scores_aligned.ndim == 1:
        scores_aligned = scores_aligned.reshape(-1, 1)
    thresholds = np.asarray(thresholds).reshape(1, -1)
    pred_each = np.where(scores_aligned > thresholds, 4.0, 1.0)
    pred_each[np.isnan(scores_aligned)] = np.nan

    any_anom = (pred_each == 4).any(axis=1)
    all_nan = np.isnan(pred_each).all(axis=1)
    out = np.where(any_anom, 4.0, 1.0)
    out = out.astype(float)
    out[all_nan] = np.nan
    out[pd.isna(original_value).values] = np.nan
    return pd.Series(out)


def metric_row(df: pd.DataFrame, split: str, exp: ExperimentItem, threshold_repr: str, plot_dir: Path, pseudo_test: bool = False, note: str = "") -> Dict:
    y_true = pd.to_numeric(df["flag_2qc"], errors="coerce")
    y_1qc_raw = pd.to_numeric(df["flag_1qc"], errors="coerce")
    y_ai_flag = pd.to_numeric(df["ai_pot_flag4"], errors="coerce")
    orig_val = pd.to_numeric(df["orig_value"], errors="coerce")

    y_1qc = np.where(y_1qc_raw.isna(), np.nan, np.where(y_1qc_raw >= 400, 1, 0))
    y_ai = np.where(np.isnan(y_ai_flag), np.nan, np.where(y_ai_flag == 4, 1, 0))
    y_comb = np.where(np.isnan(y_1qc) | np.isnan(y_ai), np.nan, np.where((y_1qc == 1) | (y_ai == 1), 1, 0))

    mask = (
        (~orig_val.isna())
        & (~y_true.isna())
        & (~pd.Series(y_1qc).isna())
        & (~pd.Series(y_ai).isna())
    )

    y_t = y_true[mask].astype(int).values
    y1 = pd.Series(y_1qc)[mask].astype(int).values
    ya = pd.Series(y_ai)[mask].astype(int).values
    yc = pd.Series(y_comb)[mask].astype(int).values

    if len(y_t) == 0:
        recall_1qc = np.nan
        f1_1qc = np.nan
        recall_ai = np.nan
        f1_ai = np.nan
        recall_comb = np.nan
        f1_comb = np.nan
        ai_usable = np.nan
    else:
        recall_1qc = float(recall_score(y_t, y1, zero_division=0))
        f1_1qc = float(f1_score(y_t, y1, zero_division=0))
        recall_ai = float(recall_score(y_t, ya, zero_division=0))
        f1_ai = float(f1_score(y_t, ya, zero_division=0))
        recall_comb = float(recall_score(y_t, yc, zero_division=0))
        f1_comb = float(f1_score(y_t, yc, zero_division=0))
        ai_usable = bool((recall_ai > recall_1qc) and (f1_ai > f1_1qc))

    return {
        "top_folder": exp.top_folder,
        "experiment_folder": exp.experiment_folder,
        "data_name": exp.data_name,
        "pair": exp.pair,
        "pot_key": exp.pot_key,
        "split": split,
        "n_total_rows": int(len(df)),
        "n_eval_rows": int(mask.sum()),
        "pot_scaler": float(exp.pot_scaler),
        "pot_threshold": threshold_repr,
        "recall_1qc": recall_1qc,
        "f1_1qc": f1_1qc,
        "recall_ai_pot": recall_ai,
        "f1_ai_pot": f1_ai,
        "recall_1qc_plus_ai": recall_comb,
        "f1_1qc_plus_ai": f1_comb,
        "ai_usable": ai_usable,
        "pseudo_test": bool(pseudo_test),
        "note": note,
        "plot_dir": str(plot_dir),
    }


def build_plot_frames(values_df: pd.DataFrame, flags2_df: pd.DataFrame, flags1_df: pd.DataFrame, ai_flag: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_col = [c for c in values_df.columns if c != DATE_COL_ALIAS][0]
    out = values_df[[DATE_COL_ALIAS, value_col]].copy()
    out = out.rename(columns={DATE_COL_ALIAS: PLOT_DATE_COL})
    out[f"{value_col}_AI_FLAG"] = ai_flag.values

    f2 = flags2_df.copy().rename(columns={DATE_COL_ALIAS: PLOT_DATE_COL})
    f1 = flags1_df.copy().rename(columns={DATE_COL_ALIAS: PLOT_DATE_COL})
    return out, f2, f1


def run_plot(values_df: pd.DataFrame, flags2_df: pd.DataFrame, flags1_df: pd.DataFrame, ai_flag: pd.Series, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    base_df, f2, f1 = build_plot_frames(values_df, flags2_df, flags1_df, ai_flag)
    plot_all_features_anomaly_with_flags(
        test_with_AI_flag=base_df,
        test_flag_df=f2,
        test_1qc_flag_df=f1,
        save_dir=str(save_dir),
    )


def run_plot_fallback(values_df: pd.DataFrame, flags2_df: pd.DataFrame, flags1_df: pd.DataFrame, ai_flag: pd.Series, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    value_col = [c for c in values_df.columns if c != DATE_COL_ALIAS][0]
    f2_cols = [c for c in flags2_df.columns if c != DATE_COL_ALIAS]
    f1_cols = [c for c in flags1_df.columns if c != DATE_COL_ALIAS]
    f2_col = f2_cols[0] if f2_cols else None
    f1_col = f1_cols[0] if f1_cols else None

    base = values_df[[DATE_COL_ALIAS, value_col]].copy()
    base["ai_flag"] = ai_flag.values
    base = base.sort_values(DATE_COL_ALIAS).reset_index(drop=True)

    merged = base[[DATE_COL_ALIAS, value_col, "ai_flag"]].copy()
    if f2_col is not None:
        merged = merged.merge(flags2_df[[DATE_COL_ALIAS, f2_col]], on=DATE_COL_ALIAS, how="left")
        merged = merged.rename(columns={f2_col: "flag2"})
    else:
        merged["flag2"] = np.nan
    if f1_col is not None:
        merged = merged.merge(flags1_df[[DATE_COL_ALIAS, f1_col]], on=DATE_COL_ALIAS, how="left")
        merged = merged.rename(columns={f1_col: "flag1"})
    else:
        merged["flag1"] = np.nan

    ds = max(1, len(merged) // 50000)
    sub = merged.iloc[::ds, :]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(sub[DATE_COL_ALIAS], sub[value_col], color="black", linewidth=0.2, marker="o", markersize=0.8, zorder=1)

    gt = merged["flag2"] == 1
    if gt.any():
        d = merged.loc[gt]
        ax.scatter(d[DATE_COL_ALIAS], d[value_col], color="red", s=4, zorder=3, label="2QC")

    q1 = pd.to_numeric(merged["flag1"], errors="coerce")
    q1a = (q1 >= 400)
    if q1a.any():
        d = merged.loc[q1a]
        ax.scatter(d[DATE_COL_ALIAS], d[value_col], edgecolors="orange", facecolors="none", marker="D", s=8, linewidths=0.7, zorder=4, label="1QC")

    ai = merged["ai_flag"] == 4
    if ai.any():
        d = merged.loc[ai]
        ax.scatter(d[DATE_COL_ALIAS], d[value_col], edgecolors="blue", facecolors="none", s=8, linewidths=0.7, zorder=5, label="AI QC")

    ax.set_title(f"Anomaly Detection (fallback): {value_col}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    safe = re.sub(r'[\\/*?:"<>|()]+', "_", value_col)
    fig.savefig(save_dir / f"anomaly_df_{safe}.png", dpi=200)
    plt.close(fig)


def prepare_manifest(
    output_bilstm_dir: Path,
    ai_models_dir: Path,
    pot_scale_map: Dict[str, float],
    top_filter: Optional[List[str]] = None,
) -> List[ExperimentItem]:
    top_folders = sorted([p for p in output_bilstm_dir.iterdir() if p.is_dir() and p.name != "ai_models"], key=lambda p: p.name)
    items = []
    for top in top_folders:
        if top_filter and top.name not in top_filter:
            continue
        data_name, pair = parse_top_folder(top.name)
        pot_key = f"{data_name}/{pair}"
        if pot_key not in pot_scale_map:
            raise KeyError(f"Missing pot scaler key: {pot_key}")
        exp_dir = select_experiment_dir(top)
        model_file = select_model_file(ai_models_dir / top.name)
        scaler_path = exp_dir / "scaler" / "minmax_scaler.joblib"
        seq_len = parse_seq_len(exp_dir.name, default_seq=8)
        items.append(
            ExperimentItem(
                top_folder=top.name,
                experiment_folder=exp_dir.name,
                experiment_path=exp_dir,
                model_path=model_file,
                scaler_path=scaler_path if scaler_path.exists() else None,
                data_name=data_name,
                pair=pair,
                pot_key=pot_key,
                pot_scaler=pot_scale_map[pot_key],
                seq_len=seq_len,
            )
        )
    return items


def evaluate_experiment(exp: ExperimentItem, df_root: Path, out_plot_root: Path, device: str) -> Tuple[List[Dict], Optional[Dict], Optional[str]]:
    try:
        train_v, train_f2, train_f1 = load_split_frames(df_root, exp.data_name, exp.pair, "train")
        test_v, test_f2, test_f1 = load_split_frames(df_root, exp.data_name, exp.pair, "test")

        value_col = [c for c in train_v.columns if c != DATE_COL_ALIAS][0]
        flag2_col = [c for c in train_f2.columns if c != DATE_COL_ALIAS][0]
        flag1_col = [c for c in train_f1.columns if c != DATE_COL_ALIAS][0]

        n_layers, hidden_size = infer_arch_from_ckpt(exp.model_path)
        ckpt = torch.load(exp.model_path, map_location=device)
        sd = ckpt["model_state_dict"]
        n_features = int(sd["encoder.lstm.weight_ih_l0"].shape[1])
        model = BiLSTMAutoEncoder(
            num_layers=n_layers,
            hidden_size=hidden_size,
            nb_feature=n_features,
            dropout=0.2,
            device=device,
        ).to(device)
        model.load_state_dict(sd)
        model.eval()

        if exp.scaler_path is not None:
            scaler = joblib.load(exp.scaler_path)
        else:
            scaler = None

        train_phys = apply_physical_limits(train_v[[DATE_COL_ALIAS, value_col]].copy())
        train_series_clean = pd.to_numeric(train_phys[value_col], errors="coerce")
        train_flag2 = pd.to_numeric(train_f2[flag2_col], errors="coerce")
        train_series_clean = train_series_clean.mask(train_flag2 == 1, np.nan)

        def to_model_matrix(series: pd.Series) -> np.ndarray:
            vals = pd.to_numeric(series, errors="coerce").values.astype(float)
            if n_features == 1:
                return vals.reshape(-1, 1)
            if n_features == 2 and "dir" in exp.pair.lower():
                rad = np.deg2rad(vals)
                u = -np.sin(rad)
                v = -np.cos(rad)
                return np.column_stack([u, v])
            raise RuntimeError(f"Unsupported feature shape for {exp.top_folder}: n_features={n_features}")

        train_clean_scaled_src = to_model_matrix(train_series_clean)

        if scaler is None:
            tmp_series = pd.Series(train_series_clean).interpolate(limit_direction="both")
            tmp = to_model_matrix(tmp_series)
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            scaler.fit(tmp)

        train_clean_scaled = scaler.transform(train_clean_scaled_src)
        train_clean_seqs = build_sequences(train_clean_scaled, exp.seq_len)
        if len(train_clean_seqs) == 0:
            raise RuntimeError("No train sequences for threshold.")
        mask_no_nan = ~np.isnan(train_clean_seqs).any(axis=(1, 2))
        train_clean_seqs = train_clean_seqs[mask_no_nan]
        train_clean_scores = score_sequences(model, train_clean_seqs, device=device)
        if len(train_clean_scores) == 0:
            raise RuntimeError("No train scores for threshold.")

        rows = []
        split_cache = {}
        for split_name, v_df, f2_df, f1_df in [
            ("train", train_v, train_f2, train_f1),
            ("test", test_v, test_f2, test_f1),
        ]:
            s_orig = pd.to_numeric(v_df[value_col], errors="coerce")
            s_model = interpolate_series(s_orig)
            s_model_mat = to_model_matrix(s_model)
            s_scaled = scaler.transform(s_model_mat)
            seqs = build_sequences(s_scaled, exp.seq_len)
            scores = score_sequences(model, seqs, device=device)

            thresholds = []
            for fidx in range(scores.shape[1]):
                dummy_label = np.zeros(len(scores), dtype=int)
                pot_res, _ = pot_eval_stable(
                    init_score=train_clean_scores[:, fidx],
                    score=scores[:, fidx],
                    label=dummy_label,
                    q=1e-5,
                    level=0.01,
                    th_scale=exp.pot_scaler,
                )
                thresholds.append(float(pot_res["threshold"]))
            thresholds = np.asarray(thresholds, dtype=float)
            aligned_scores = align_scores(scores, len(v_df), exp.seq_len)
            ai_flag = build_ai_flag(aligned_scores, thresholds, s_orig)

            eval_df = pd.DataFrame(
                {
                    DATE_COL_ALIAS: v_df[DATE_COL_ALIAS],
                    "orig_value": s_orig,
                    "flag_2qc": pd.to_numeric(f2_df[flag2_col], errors="coerce"),
                    "flag_1qc": pd.to_numeric(f1_df[flag1_col], errors="coerce"),
                    "ai_pot_flag4": ai_flag,
                }
            )
            plot_dir = out_plot_root / exp.top_folder / split_name / "fig_pot"
            try:
                run_plot(v_df, f2_df, f1_df, ai_flag, plot_dir)
            except Exception:
                run_plot_fallback(v_df, f2_df, f1_df, ai_flag, plot_dir)
            th_repr = "|".join([f"{t:.10g}" for t in thresholds.tolist()])
            rows.append(metric_row(eval_df, split_name, exp, th_repr, plot_dir))
            split_cache[split_name] = (v_df, f2_df, f1_df, thresholds)

        pseudo_row = None
        if exp.data_name in {"SMB3_WATER", "SMB4_WATER"} and exp.pair in PSEUDO_TARGETS:
            v_all = (
                pd.concat([train_v, test_v], ignore_index=True)
                .drop_duplicates(subset=[DATE_COL_ALIAS], keep="first")
                .sort_values(DATE_COL_ALIAS)
            )
            f2_all = (
                pd.concat([train_f2, test_f2], ignore_index=True)
                .drop_duplicates(subset=[DATE_COL_ALIAS], keep="first")
                .sort_values(DATE_COL_ALIAS)
            )
            f1_all = (
                pd.concat([train_f1, test_f1], ignore_index=True)
                .drop_duplicates(subset=[DATE_COL_ALIAS], keep="first")
                .sort_values(DATE_COL_ALIAS)
            )

            m = (v_all[DATE_COL_ALIAS] >= PSEUDO_START) & (v_all[DATE_COL_ALIAS] <= PSEUDO_END)
            v_ps = v_all.loc[m].copy()
            f2_ps = f2_all[f2_all[DATE_COL_ALIAS].isin(v_ps[DATE_COL_ALIAS])].copy()
            f1_ps = f1_all[f1_all[DATE_COL_ALIAS].isin(v_ps[DATE_COL_ALIAS])].copy()
            v_ps = v_ps.sort_values(DATE_COL_ALIAS).reset_index(drop=True)
            f2_ps = f2_ps.sort_values(DATE_COL_ALIAS).reset_index(drop=True)
            f1_ps = f1_ps.sort_values(DATE_COL_ALIAS).reset_index(drop=True)

            if len(v_ps) > 0:
                s_orig = pd.to_numeric(v_ps[value_col], errors="coerce")
                s_model = interpolate_series(s_orig)
                s_model_mat = to_model_matrix(s_model)
                s_scaled = scaler.transform(s_model_mat)
                seqs = build_sequences(s_scaled, exp.seq_len)
                scores = score_sequences(model, seqs, device=device)
                thresholds = split_cache["test"][3]
                aligned_scores = align_scores(scores, len(v_ps), exp.seq_len)
                ai_flag = build_ai_flag(aligned_scores, thresholds, s_orig)
                eval_df = pd.DataFrame(
                    {
                        DATE_COL_ALIAS: v_ps[DATE_COL_ALIAS],
                        "orig_value": s_orig,
                        "flag_2qc": pd.to_numeric(f2_ps[flag2_col], errors="coerce"),
                        "flag_1qc": pd.to_numeric(f1_ps[flag1_col], errors="coerce"),
                        "ai_pot_flag4": ai_flag,
                    }
                )
                plot_dir = out_plot_root / exp.top_folder / "pseudo_test_2024" / "fig_pot"
                try:
                    run_plot(v_ps, f2_ps, f1_ps, ai_flag, plot_dir)
                except Exception:
                    run_plot_fallback(v_ps, f2_ps, f1_ps, ai_flag, plot_dir)
                th_repr = "|".join([f"{t:.10g}" for t in np.asarray(thresholds).tolist()])
                pseudo_row = metric_row(
                    eval_df,
                    split="pseudo_test_2024",
                    exp=exp,
                    threshold_repr=th_repr,
                    plot_dir=plot_dir,
                    pseudo_test=True,
                    note="Evaluation window is within training-used period (non-independent validation).",
                )
        return rows, pseudo_row, None
    except Exception:
        return [], None, traceback.format_exc()


def render_markdown_report(
    metrics_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if "split" not in metrics_df.columns:
        metrics_df = pd.DataFrame(
            columns=[
                "top_folder",
                "split",
                "n_eval_rows",
                "recall_1qc",
                "f1_1qc",
                "recall_ai_pot",
                "f1_ai_pot",
                "recall_1qc_plus_ai",
                "f1_1qc_plus_ai",
                "ai_usable",
                "plot_dir",
            ]
        )
    lines = []
    lines.append("# BiLSTM Re-Inference Evaluation Summary")
    lines.append("")
    lines.append("## Methodology")
    lines.append("- Re-inference executed from model checkpoints in `output/bilstm/ai_models`.")
    lines.append("- POT threshold computed by `pot_eval_stable` with scaler from `codex_scr/pot_configs.json` (`data_name/pair` key).")
    lines.append("- Metrics: Recall, F1 for `1QC`, `AI POT`, `1QC+AI(POT)` against 2QC target.")
    lines.append("- Exclusion rule: original value NaN OR any of 2QC/1QC/AI_POT NaN.")
    lines.append("- Qualitative outputs: POT-only plots from `plot_all_features_anomaly_with_flags`.")
    lines.append("")

    total_exp = metrics_df["top_folder"].nunique() if len(metrics_df) else 0
    usable = metrics_df[metrics_df["split"].isin(["train", "test"]) & (metrics_df["ai_usable"] == True)]
    lines.append("## Run Summary")
    lines.append(f"- Processed experiments: {total_exp}")
    lines.append(f"- Valid train/test rows: {len(metrics_df[metrics_df['split'].isin(['train','test'])])}")
    lines.append(f"- AI usable rows (`recall_ai_pot > recall_1qc` and `f1_ai_pot > f1_1qc`): {len(usable)}")
    lines.append(f"- Failed experiments: {failed_df['top_folder'].nunique() if len(failed_df) else 0}")
    lines.append("")

    lines.append("## Per-Experiment Quantitative Table (Train/Test)")
    cols = [
        "top_folder",
        "split",
        "n_eval_rows",
        "recall_1qc",
        "f1_1qc",
        "recall_ai_pot",
        "f1_ai_pot",
        "recall_1qc_plus_ai",
        "f1_1qc_plus_ai",
        "ai_usable",
    ]
    if len(metrics_df):
        table_df = metrics_df[metrics_df["split"].isin(["train", "test"])][cols].copy()
        lines.append(table_df.to_csv(index=False))
    else:
        lines.append("No train/test metrics.")
    lines.append("")

    lines.append("## Pseudo-Test (SMB3/SMB4 Water Features, 2024-01-01 ~ 2024-12-31)")
    lines.append("- Caveat: Evaluation window is within training-used period (non-independent validation).")
    if len(pseudo_df):
        pcols = [
            "top_folder",
            "split",
            "n_eval_rows",
            "recall_1qc",
            "f1_1qc",
            "recall_ai_pot",
            "f1_ai_pot",
            "recall_1qc_plus_ai",
            "f1_1qc_plus_ai",
            "ai_usable",
            "note",
        ]
        lines.append(pseudo_df[pcols].to_csv(index=False))
    else:
        lines.append("No pseudo-test rows.")
    lines.append("")

    lines.append("## Plot Index")
    if len(metrics_df):
        pi = metrics_df[["top_folder", "split", "plot_dir"]].copy()
        lines.append(pi.to_csv(index=False))
    else:
        lines.append("No plots.")
    lines.append("")

    lines.append("## Manifest Reference")
    lines.append(f"- Rows: {len(manifest_df)}")
    lines.append("- File: `output/bilstm_codex/manifests/experiment_manifest.csv`")
    lines.append("")

    if len(failed_df):
        lines.append("## Failures")
        lines.append(failed_df[["top_folder", "stage", "reason"]].to_csv(index=False))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="BiLSTM re-inference evaluation runner")
    parser.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dry-two", action="store_true", help="Run only SMB1_AIR_solar and SMB3_WATER_temp")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.base_dir)
    output_bilstm = base / "output" / "bilstm"
    ai_models = output_bilstm / "ai_models"
    df_root = base / "data_for_AI" / "df_univariate"
    out_root = base / "output" / "bilstm_codex"
    out_metrics = out_root / "metrics"
    out_reports = out_root / "reports"
    out_manifests = out_root / "manifests"
    out_plots = out_root / "plots"
    for p in [out_metrics, out_reports, out_manifests, out_plots]:
        p.mkdir(parents=True, exist_ok=True)

    pot_map = load_pot_scales(base / "codex_scr" / "pot_configs.json")
    top_filter = ["SMB1_AIR_solar", "SMB3_WATER_temp"] if args.dry_two else None
    manifest = prepare_manifest(output_bilstm, ai_models, pot_map, top_filter=top_filter)
    manifest_df = pd.DataFrame([vars(m) for m in manifest])
    manifest_df["experiment_path"] = manifest_df["experiment_path"].astype(str)
    manifest_df["model_path"] = manifest_df["model_path"].astype(str)
    manifest_df["scaler_path"] = manifest_df["scaler_path"].astype(str)
    manifest_df.to_csv(out_manifests / "experiment_manifest.csv", index=False, encoding="utf-8-sig")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows: List[Dict] = []
    pseudo_rows: List[Dict] = []
    failures: List[Dict] = []

    for i, exp in enumerate(manifest, start=1):
        print(f"[{i}/{len(manifest)}] {exp.top_folder} -> {exp.experiment_folder}")
        rows, pseudo, err = evaluate_experiment(exp, df_root=df_root, out_plot_root=out_plots, device=device)
        if err is not None:
            failures.append(
                {
                    "top_folder": exp.top_folder,
                    "experiment_folder": exp.experiment_folder,
                    "stage": "evaluate_experiment",
                    "reason": err.splitlines()[-1] if err else "Unknown",
                    "traceback": err,
                }
            )
            continue
        metric_rows.extend(rows)
        if pseudo is not None:
            pseudo_rows.append(pseudo)

    metrics_df = pd.DataFrame(metric_rows)
    pseudo_df = pd.DataFrame(pseudo_rows)
    failed_df = pd.DataFrame(failures)

    metrics_df.to_csv(out_metrics / "metrics_train_test_all.csv", index=False, encoding="utf-8-sig")
    pseudo_df.to_csv(out_metrics / "metrics_pseudo_test_smb3_smb4.csv", index=False, encoding="utf-8-sig")
    failed_df.to_csv(out_metrics / "missing_or_failed_experiments.csv", index=False, encoding="utf-8-sig")
    render_markdown_report(metrics_df, pseudo_df, failed_df, manifest_df, out_reports / "evaluation_summary.md")

    print("Done.")
    print(f"manifest: {out_manifests / 'experiment_manifest.csv'}")
    print(f"metrics : {out_metrics / 'metrics_train_test_all.csv'}")
    print(f"pseudo  : {out_metrics / 'metrics_pseudo_test_smb3_smb4.csv'}")
    print(f"failed  : {out_metrics / 'missing_or_failed_experiments.csv'}")
    print(f"report  : {out_reports / 'evaluation_summary.md'}")


if __name__ == "__main__":
    main()
