# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 13:33:33 2025

@author: user
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
BASE_DIR = Path(__file__).resolve().parent

# ?꾩옱源뚯????ㅽ????ㅼ젙??珥덇린??
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False # 留덉씠?덉뒪 遺???쒖떆 臾몄젣瑜??닿껐
try:
    local_font = BASE_DIR / "assets" / "fonts" / "NanumGothic.ttf"
    if local_font.exists():
        font = fm.FontProperties(fname=str(local_font))
        plt.rc('font', family=font.get_name())
except Exception:
    pass


#%%
def safe_filename(name):
    # ?뚯씪紐낆뿉???덉슜?섏? ?딅뒗 臾몄옄 ?쒓굅
    name = re.sub(r'[\\/*?:"<>|]', '_', name)  # Windows forbidden characters
    name = name.replace("(", "_").replace(")", "_").replace("/", "_")
    return name

def plot_anomaly_with_2qc(
    test_values_df,
    test_scores_np,
    threshold,
    seq_len,
    test_flag_df=None,
    feature_name=None,
    feature_index=None,
    save_path=None
):

    time = pd.to_datetime(test_values_df["?좎쭨"])

    if feature_name is None:
        feature_name = test_values_df.columns[1]
    values = test_values_df[feature_name].values


    # ------------------------------------
    # 1) anomaly score align
    # ------------------------------------
    N = len(values)
    scores_aligned = np.full(N, np.nan)
    valid_idx = []

    for i, score in enumerate(test_scores_np):
        idx = seq_len - 1 + i
        if idx < N:
            scores_aligned[idx] = score
            valid_idx.append(idx)

    valid_idx = np.array(valid_idx, dtype=int)



    # ------------------------------------
    # 2) GT flag == 1 ?꾩튂 李얘린
    # ------------------------------------
    gt_idx = None
    if test_flag_df is not None:
        if feature_index is not None:
            gt_flag = test_flag_df.iloc[:, feature_index].values
        gt_idx = np.where(gt_flag == 1)[0]


    # ------------------------------------
    # 3) Plotting
    # ------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(time, values, color="black", marker='o', markersize=1, label=feature_name, linewidth=0.1, zorder=1)


    # ------------------------------------
    # ??Human QC (鍮④컙 ?ㅻえ)
    # ------------------------------------
    if gt_idx is not None and len(gt_idx) > 0:
        ax1.scatter(
            time.iloc[gt_idx],
            values[gt_idx],
            color="red",
            s=2,
            marker="o",
            zorder=3,
            label="Human QC (2QC)"
        )


    # ------------------------------------
    # ??AI anomaly: score > threshold ???뚮? X
    # ------------------------------------
    ai_idx = valid_idx[test_scores_np > threshold]

    if len(ai_idx) > 0:
        ax1.scatter(
            time.iloc[ai_idx],
            values[ai_idx],
            color="blue",
            s=10,
            marker="o",
            facecolors="none",
            linewidths=0.5,
            zorder=4,
            label="AI QC"
        )


    # ------------------------------------
    # Score axis
    # ------------------------------------
    ax2 = ax1.twinx()
    ax2.plot(time, scores_aligned, color="orange", linestyle="-", label="Anomaly Score")
    ax2.axhline(threshold, color="blue", linestyle=":", label=f"Threshold ({threshold:.4f})")
    ax2.set_ylabel("Anomaly Score")

    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, ymax * 2)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title(f"Anomaly Detection: {feature_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] Saved plot: {save_path}")

    plt.show()


def plot_all_features(
    test_values_df,
    test_flag_df,
    test_scores_dict,
    pot_results_dict,
    seq_len,
    save_dir
):

    features = [c for c in test_values_df.columns if c != "?좎쭨"]

    for f_idx, feature_name in enumerate(features):

        file_feature_name = safe_filename(feature_name)

        test_scores_np = test_scores_dict[f_idx]
        threshold = pot_results_dict[f_idx]["threshold"]

        # ??save_path?쇰뒗 蹂꾨룄 蹂?섎줈 ?뚯씪寃쎈줈 ?앹꽦 (save_dir 蹂댄샇)
        save_path = os.path.join(save_dir, f"anomaly_{file_feature_name}.png")

        print(f"\n[INFO] Plotting Feature {f_idx}: {feature_name}")
        print(f"       test_scores len = {len(test_scores_np)}")
        print(f"       threshold = {threshold}")

        plot_anomaly_with_2qc(
            test_values_df=test_values_df,
            test_scores_np=test_scores_np,
            threshold=threshold,
            seq_len=seq_len,
            test_flag_df=test_flag_df,
            feature_name=feature_name,
            feature_index=f_idx, 
            save_path=save_path      # ???щ컮瑜닿쾶 ?뚯씪 寃쎈줈留??꾨떖
        )
#%%
def find_ai_flag_column(test_df, feature_name):
    """
    feature_name怨?媛???좎궗??AI_FLAG 而щ읆 ?먮룞 ?먯깋
    ?? feature_name='SMB1_?랁뼢(deg)'
        ?ㅼ젣 而щ읆='SMB1_?랁뼢_AI_FLAG'
    """
    # ?꾨낫 = '_AI_FLAG'濡??앸굹??而щ읆??
    ai_cols = [c for c in test_df.columns if c.endswith("_AI_FLAG")]

    # 愿꾪샇, 怨듬갚 ?쒓굅
    clean_feature = (
        feature_name.replace("(deg)", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "")
    )

    for c in ai_cols:
        clean_c = (
            c.replace("_AI_FLAG", "")
             .replace("(deg)", "")
             .replace("(", "")
             .replace(")", "")
             .replace(" ", "")
        )
        
        if clean_c == clean_feature:
            return c

    # 紐살갼?쇰㈃ None
    return None

def plot_all_features_anomaly_with_flags(
    test_with_AI_flag,
    test_flag_df=None,
    test_1qc_flag_df=None,
    save_dir=None
):
    # ---------------------------------------
    # 湲곗? DF (AI 寃곌낵) ?쒓컙異?
    # ---------------------------------------
    base_df = test_with_AI_flag.copy()
    base_df["?좎쭨"] = pd.to_datetime(base_df["?좎쭨"])

    # ---------------------------------------
    # feature 紐⑸줉
    # ---------------------------------------
    features = [
        c for c in base_df.columns
        if c != "?좎쭨"
        and not c.endswith("_AI_FLAG")
        and not c.startswith("FLAG_")
    ]

    # ---------------------------------------
    # GT / 1QC DF ?좎쭨 ?뺣젹
    # ---------------------------------------
    if test_flag_df is not None:
        test_flag_df = test_flag_df.copy()
        test_flag_df["?좎쭨"] = pd.to_datetime(test_flag_df["?좎쭨"])

    if test_1qc_flag_df is not None:
        test_1qc_flag_df = test_1qc_flag_df.copy()
        test_1qc_flag_df["?좎쭨"] = pd.to_datetime(test_1qc_flag_df["?좎쭨"])

    # ---------------------------------------
    # feature loop
    # ---------------------------------------
    for feature_name in features:
        
        print(f"\n[INFO] Plotting feature: {feature_name}")
    
        time = base_df["?좎쭨"]
        values = base_df[feature_name].values
    
        # ------------------------------------
        # GT (2QC) ???좎쭨 湲곗? merge
        # ------------------------------------
        gt_idx = None
        if test_flag_df is not None:

            if feature_name in ('Sea_level_Repr_Sea', 'Sea_level_Repr_Lake'):
                gt_col = f"Sea_level_FLAG_{feature_name.split('_')[2]}_{feature_name.split('_')[3]}"
            
            else:
                gt_col = f"{feature_name.split('_')[0]}_FLAG_{feature_name.split('_', 1)[1]}"

    
            if gt_col in test_flag_df.columns:
                merged_gt = base_df[["?좎쭨"]].merge(
                    test_flag_df[["?좎쭨", gt_col]],
                    on="?좎쭨",
                    how="left"
                )
                gt_idx = np.where(merged_gt[gt_col] == 1)[0]
    
        # ------------------------------------
        # 1QC ???좎쭨 湲곗? merge
        # ------------------------------------
        qc1_idx_3xx = qc1_idx_402 = qc1_idx_4xx = None
    
        if test_1qc_flag_df is not None:
            if feature_name in ('Sea_level_Repr_Sea', 'Sea_level_Repr_Lake'):
                qc1_col = "waterlevel_qc_result"
            
            else:
                qc1_col = f"{feature_name.split('_')[0]}_FLAG_{feature_name.split('_', 1)[1]}"
        

            if qc1_col in test_1qc_flag_df.columns:
                merged_qc1 = base_df[["?좎쭨"]].merge(
                    test_1qc_flag_df[["?좎쭨", qc1_col]],
                    on="?좎쭨",
                    how="left"
                )
    
                qc1_flag = pd.to_numeric(
                    merged_qc1[qc1_col],
                    errors="coerce"
                ).values
    
                qc1_idx_3xx = np.where((qc1_flag >= 300) & (qc1_flag < 400))[0]
                qc1_idx_402 = np.where(qc1_flag == 402)[0]
                qc1_idx_4xx = np.where(
                    (qc1_flag >= 400) & (qc1_flag < 500) & (qc1_flag != 402)
                )[0]
    
        # ------------------------------------
        # AI FLAG (媛숈? DF??index OK)
        # ------------------------------------
        ai_col = find_ai_flag_column(base_df, feature_name)
        ai_idx = None
        if ai_col is not None:
            ai_idx = np.where(base_df[ai_col].values == 4)[0]
    
        # ------------------------------------
        # Plot
        # ------------------------------------
        fig, ax = plt.subplots(figsize=(16, 6))
    
        ax.plot(
            time, values,
            color="black",
            marker="o",
            markersize=1,
            linewidth=0.1,
            zorder=1
        )
    
        # ---- 2QC ----
        if gt_idx is not None and len(gt_idx) > 0:
            ax.scatter(
                time.iloc[gt_idx],
                values[gt_idx],
                color="red",
                s=4,
                zorder=2,
                label="2QC"
            )
    
        # ---- 1QC : 3XX ----
        if qc1_idx_3xx is not None and len(qc1_idx_3xx) > 0:
            ax.scatter(
                time.iloc[qc1_idx_3xx],
                values[qc1_idx_3xx],
                marker="D",
                edgecolors="gray",
                facecolors="none",
                s=11,
                linewidths=1,
                zorder=3,
                label="1QC (3)"
            )
    
        # ---- 1QC : 4XX (402 ?쒖쇅) ----
        if qc1_idx_4xx is not None and len(qc1_idx_4xx) > 0:
            ax.scatter(
                time.iloc[qc1_idx_4xx],
                values[qc1_idx_4xx],
                marker="D",
                edgecolors="orange",
                facecolors="none",
                s=13,
                linewidths=1.2,
                zorder=4,
                label="1QC (4)"
            )
    
        # ---- 1QC : 402 ----
        if qc1_idx_402 is not None and len(qc1_idx_402) > 0:
            ax.scatter(
                time.iloc[qc1_idx_402],
                values[qc1_idx_402],
                color="green",
                s=4,
                zorder=3,
                label="1QC (loc)"
            )
    
        # ---- AI QC ----
        if ai_idx is not None and len(ai_idx) > 0:
            ax.scatter(
                time.iloc[ai_idx],
                values[ai_idx],
                color="blue",
                facecolors="none",
                s=6,
                linewidths=0.6,
                zorder=5,
                label="AI QC"
            )
    
        ax.set_title(f"Anomaly Detection: {feature_name}")
        ax.legend(loc="upper left")
        plt.tight_layout()

        if save_dir is not None:
            save_path = os.path.join(
                save_dir, f"anomaly_df_{safe_filename(feature_name)}.png"
            )
            plt.savefig(save_path, dpi=200)
            print(f"[INFO] Saved plot: {save_path}")

        plt.show()

#%%

def plot_all_features_anomaly_with_flags_custom(
    test_with_AI_flag,
    test_flag_df,
    test_1qc_flag_df,
    start_date,
    end_date,
    save_dir=None
):
    # ---------------------------------------
    # 湲곗? DF (AI 寃곌낵) ?쒓컙異?
    # ---------------------------------------
    base_df = test_with_AI_flag.copy()
    base_df["?좎쭨"] = pd.to_datetime(base_df["?좎쭨"])
    
    # ===== ?좎쭨 ?꾪꽣 =====
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    if start_date is not None or end_date is not None:
        mask = pd.Series(True, index=base_df.index)
    
        if start_date is not None:
            mask &= base_df["?좎쭨"] >= start_date
        if end_date is not None:
            mask &= base_df["?좎쭨"] <= end_date
    
        base_df = base_df.loc[mask].reset_index(drop=True)


    # ---------------------------------------
    # feature 紐⑸줉
    # ---------------------------------------
    features = [
        c for c in base_df.columns
        if c != "?좎쭨"
        and not c.endswith("_AI_FLAG")
        and not c.startswith("FLAG_")
    ]

    # ---------------------------------------
    # GT / 1QC DF ?좎쭨 ?뺣젹
    # ---------------------------------------
    if test_flag_df is not None:
        test_flag_df = test_flag_df.copy()
        test_flag_df["?좎쭨"] = pd.to_datetime(test_flag_df["?좎쭨"])

    if test_1qc_flag_df is not None:
        test_1qc_flag_df = test_1qc_flag_df.copy()
        test_1qc_flag_df["?좎쭨"] = pd.to_datetime(test_1qc_flag_df["?좎쭨"])

    # ---------------------------------------
    # feature loop
    # ---------------------------------------
    for feature_name in features:
        
        print(f"\n[INFO] Plotting feature: {feature_name}")
    
        time = base_df["?좎쭨"]
        values = base_df[feature_name].values
    
        # ------------------------------------
        # GT (2QC) ???좎쭨 湲곗? merge
        # ------------------------------------
        gt_idx = None
        if test_flag_df is not None:

            if feature_name in ('Sea_level_Repr_Sea', 'Sea_level_Repr_Lake'):
                gt_col = f"Sea_level_FLAG_{feature_name.split('_')[2]}_{feature_name.split('_')[3]}"
            
            else:
                gt_col = f"{feature_name.split('_')[0]}_FLAG_{feature_name.split('_', 1)[1]}"

    
            if gt_col in test_flag_df.columns:
                merged_gt = base_df[["?좎쭨"]].merge(
                    test_flag_df[["?좎쭨", gt_col]],
                    on="?좎쭨",
                    how="left"
                )
                gt_idx = np.where(merged_gt[gt_col] == 1)[0]
        
        # ------------------------------------
        # 1QC ???좎쭨 湲곗? merge
        # ------------------------------------
        qc1_idx_3xx = qc1_idx_402 = qc1_idx_4xx = None
    
        if test_1qc_flag_df is not None:
            if feature_name in ('Sea_level_Repr_Sea', 'Sea_level_Repr_Lake'):
                qc1_col = "waterlevel_qc_result"
            
            else:
                qc1_col = f"{feature_name.split('_')[0]}_FLAG_{feature_name.split('_', 1)[1]}"
    
            if qc1_col in test_1qc_flag_df.columns:
                merged_qc1 = base_df[["?좎쭨"]].merge(
                    test_1qc_flag_df[["?좎쭨", qc1_col]],
                    on="?좎쭨",
                    how="left"
                )
    
                qc1_flag = pd.to_numeric(
                    merged_qc1[qc1_col],
                    errors="coerce"
                ).values
    
                qc1_idx_3xx = np.where((qc1_flag >= 300) & (qc1_flag < 400))[0]
                qc1_idx_402 = np.where(qc1_flag == 402)[0]
                qc1_idx_4xx = np.where(
                    (qc1_flag >= 400) & (qc1_flag < 500) & (qc1_flag != 402)
                )[0]
    
        # ------------------------------------
        # AI FLAG (媛숈? DF??index OK)
        # ------------------------------------
        ai_col = find_ai_flag_column(base_df, feature_name)
        ai_idx = None
        if ai_col is not None:
            ai_idx = np.where(base_df[ai_col].values == 4)[0]
    
        # ------------------------------------
        # Plot
        # ------------------------------------
        fig, ax = plt.subplots(figsize=(14, 5))
    
        ax.plot(
            time, values,
            color="black",
            marker="o",
            markersize=2,
            linewidth=0.1,
            zorder=1,
            label="raw"
        )
    
        # ---- 2QC ----
        if gt_idx is not None and len(gt_idx) > 0:
            ax.scatter(
                time.iloc[gt_idx],
                values[gt_idx],
                color="red",
                s=20,
                zorder=2,
                label="2QC"
            )
    
        # ---- 1QC : 3XX ----
        if qc1_idx_3xx is not None and len(qc1_idx_3xx) > 0:
            ax.scatter(
                time.iloc[qc1_idx_3xx],
                values[qc1_idx_3xx],
                marker="D",
                edgecolors="gray",
                facecolors="none",
                s=40,
                linewidths=1,
                zorder=3,
                label="1QC (3)"
            )
    
        # ---- 1QC : 4XX (402 ?쒖쇅) ----
        if qc1_idx_4xx is not None and len(qc1_idx_4xx) > 0:
            ax.scatter(
                time.iloc[qc1_idx_4xx],
                values[qc1_idx_4xx],
                marker="D",
                edgecolors="orange",
                facecolors="none",
                s=40,
                linewidths=1.2,
                zorder=4,
                label="1QC (4)"
            )
    
        # ---- 1QC : 402 ----
        if qc1_idx_402 is not None and len(qc1_idx_402) > 0:
            ax.scatter(
                time.iloc[qc1_idx_402],
                values[qc1_idx_402],
                color="green",
                s=20,
                zorder=3,
                label="1QC (4-loc)"
            )
    
        # ---- AI QC ----
        if ai_idx is not None and len(ai_idx) > 0:
            ax.scatter(
                time.iloc[ai_idx],
                values[ai_idx],
                color="blue",
                facecolors="none",
                s=22,
                linewidths=1,
                zorder=5,
                label="AI QC"
            )
    
        ax.set_title(f"Anomaly Detection: {feature_name}")
        ax.legend(loc="upper left")
        plt.tight_layout()
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str   = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        if save_dir is not None:
            save_path = os.path.join(
                save_dir, f"anomaly_df_{safe_filename(feature_name)}_custom_{start_str}_{end_str}.png"
            )
            plt.savefig(save_path, dpi=200)
            print(f"[INFO] Saved plot: {save_path}")

        plt.show()
    
#%% 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _build_windows_from_indices(idxs, n, pre=5, post=5):
    """
    idxs: anomaly indices (array-like)
    n: total length
    return: merged windows list of (start, end) inclusive
    """
    if idxs is None or len(idxs) == 0:
        return []

    idxs = np.unique(np.asarray(idxs, dtype=int))
    windows = []
    for i in idxs:
        s = max(0, i - pre)
        e = min(n - 1, i + post)
        windows.append((s, e))

    windows.sort()
    merged = [windows[0]]
    for s, e in windows[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:  # overlap or adjacent
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def plot_ai_window(
    test_with_AI_flag,
    test_flag_df,
    test_1qc_flag_df,
    start_date,
    end_date,
    save_dir=None,
    pre_n=5,
    post_n=5,
    skip_no_ai=True
):
    """
    AI_FLAG==4媛 ?섏삩 吏?먯쓣 湲곗??쇰줈 ?욌뮘(pre_n, post_n)留??뺣??댁꽌 window蹂꾨줈 ?뚮’.
    """

    # ---------------------------------------
    # 0) Base DF + ?좎쭨 ?꾪꽣
    # ---------------------------------------
    base_df = test_with_AI_flag.copy()
    base_df["?좎쭨"] = pd.to_datetime(base_df["?좎쭨"])

    start_date = pd.to_datetime(start_date) if start_date is not None else None
    end_date   = pd.to_datetime(end_date) if end_date is not None else None

    if start_date is not None or end_date is not None:
        mask = pd.Series(True, index=base_df.index)
        if start_date is not None:
            mask &= base_df["?좎쭨"] >= start_date
        if end_date is not None:
            mask &= base_df["?좎쭨"] <= end_date
        base_df = base_df.loc[mask].reset_index(drop=True)

    if len(base_df) == 0:
        print("[WARN] base_df is empty after date filtering.")
        return

    # ---------------------------------------
    # 1) feature 紐⑸줉
    # ---------------------------------------
    features = [
        c for c in base_df.columns
        if c != "?좎쭨"
        and not c.endswith("_AI_FLAG")
        and not c.startswith("FLAG_")
    ]

    # ---------------------------------------
    # 2) GT / 1QC DF ?좎쭨 ?뺣━
    # ---------------------------------------
    if test_flag_df is not None:
        test_flag_df = test_flag_df.copy()
        test_flag_df["?좎쭨"] = pd.to_datetime(test_flag_df["?좎쭨"])

    if test_1qc_flag_df is not None:
        test_1qc_flag_df = test_1qc_flag_df.copy()
        test_1qc_flag_df["?좎쭨"] = pd.to_datetime(test_1qc_flag_df["?좎쭨"])

    # ????대뜑
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------
    # 3) feature loop
    # ---------------------------------------
    for feature_name in features:
        print(f"\n[INFO] Feature: {feature_name}")

        # -----------------------------
        # (A) AI FLAG index 李얘린
        # -----------------------------
        ai_col = find_ai_flag_column(base_df, feature_name)
        if ai_col is None or ai_col not in base_df.columns:
            print(f"[SKIP] AI flag column not found for {feature_name}")
            continue

        ai_idx = np.where(base_df[ai_col].values == 4)[0]

        if (ai_idx is None or len(ai_idx) == 0) and skip_no_ai:
            print(f"[SKIP] No AI anomalies for {feature_name}")
            continue

        # -----------------------------
        # (B) window ?앹꽦 (짹N) + merge
        # -----------------------------
        n_total = len(base_df)
        windows = _build_windows_from_indices(ai_idx, n_total, pre=pre_n, post=post_n)

        # AI媛 ?녾퀬 ?ㅽ궢 ???섎㈃ ?꾩껜 援ш컙 1??
        if len(windows) == 0:
            windows = [(0, n_total - 1)]

        # -----------------------------
        # (C) GT 而щ읆紐?寃곗젙
        # -----------------------------
        gt_col = None
        if test_flag_df is not None:
            if feature_name in ("Sea_level_Repr_Sea", "Sea_level_Repr_Lake"):
                # Sea_level_Repr_Sea -> Sea_level_FLAG_Repr_Sea ?뺥깭媛 ?꾨땲??
                # ??湲곗〈 洹쒖튃??洹몃?濡??좎?
                gt_col = f"Sea_level_FLAG_{feature_name.split('_')[2]}_{feature_name.split('_')[3]}"
            else:
                gt_col = f"{feature_name.split('_')[0]}_FLAG_{feature_name.split('_', 1)[1]}"

        # -----------------------------
        # (D) 1QC 而щ읆紐?寃곗젙
        # -----------------------------
        qc1_col = None
        if test_1qc_flag_df is not None:
            if feature_name in ("Sea_level_Repr_Sea", "Sea_level_Repr_Lake"):
                qc1_col = "waterlevel_qc_result"
            else:
                qc1_col = f"{feature_name.split('_')[0]}_FLAG_{feature_name.split('_', 1)[1]}"

        # -----------------------------
        # (E) window蹂?plot
        # -----------------------------
        for w_i, (ws, we) in enumerate(windows, start=1):
            sub_df = base_df.iloc[ws:we+1].reset_index(drop=True)

            time = sub_df["?좎쭨"]
            values = sub_df[feature_name].values

            # ---- 2QC idx (sub_df 湲곗?)
            gt_idx = None
            if test_flag_df is not None and gt_col in test_flag_df.columns:
                merged_gt = sub_df[["?좎쭨"]].merge(
                    test_flag_df[["?좎쭨", gt_col]],
                    on="?좎쭨",
                    how="left"
                )
                gt_idx = np.where(merged_gt[gt_col] == 1)[0]

            # ---- 1QC idx (sub_df 湲곗?)
            qc1_idx_3xx = qc1_idx_402 = qc1_idx_4xx = None
            if test_1qc_flag_df is not None and qc1_col in test_1qc_flag_df.columns:
                merged_qc1 = sub_df[["?좎쭨"]].merge(
                    test_1qc_flag_df[["?좎쭨", qc1_col]],
                    on="?좎쭨",
                    how="left"
                )
                qc1_flag = pd.to_numeric(merged_qc1[qc1_col], errors="coerce").values

                qc1_idx_3xx = np.where((qc1_flag >= 300) & (qc1_flag < 400))[0]
                qc1_idx_402 = np.where(qc1_flag == 402)[0]
                qc1_idx_4xx = np.where((qc1_flag >= 400) & (qc1_flag < 500) & (qc1_flag != 402))[0]

            # ---- AI idx (sub_df 湲곗?)
            ai_idx_sub = np.where(sub_df[ai_col].values == 4)[0]

            # ---- Plot
            fig, ax = plt.subplots(figsize=(14, 5))

            ax.plot(
                time, values,
                color="black",
                marker="o",
                markersize=2,
                linewidth=0.1,
                zorder=1,
                label="raw"
            )

            if gt_idx is not None and len(gt_idx) > 0:
                ax.scatter(time.iloc[gt_idx], values[gt_idx],
                           color="red", s=20, zorder=2, label="2QC")

            if qc1_idx_3xx is not None and len(qc1_idx_3xx) > 0:
                ax.scatter(time.iloc[qc1_idx_3xx], values[qc1_idx_3xx],
                           marker="D", edgecolors="gray", facecolors="none",
                           s=40, linewidths=1, zorder=3, label="1QC (3xx)")

            if qc1_idx_4xx is not None and len(qc1_idx_4xx) > 0:
                ax.scatter(time.iloc[qc1_idx_4xx], values[qc1_idx_4xx],
                           marker="D", edgecolors="orange", facecolors="none",
                           s=40, linewidths=1.2, zorder=4, label="1QC (4xx)")

            if qc1_idx_402 is not None and len(qc1_idx_402) > 0:
                ax.scatter(time.iloc[qc1_idx_402], values[qc1_idx_402],
                           color="green", s=20, zorder=3, label="1QC (402)")

            if ai_idx_sub is not None and len(ai_idx_sub) > 0:
                ax.scatter(time.iloc[ai_idx_sub], values[ai_idx_sub],
                           color="blue", facecolors="none",
                           s=22, linewidths=1, zorder=5, label="AI QC")

            # ?쒕ぉ/踰붿쐞
            t0 = time.iloc[0].strftime("%Y-%m-%d %H:%M")
            t1 = time.iloc[-1].strftime("%Y-%m-%d %H:%M")
            ax.set_title(f"{feature_name} | AI window {w_i}/{len(windows)} | {t0} ~ {t1}")
            ax.legend(loc="upper left")
            plt.tight_layout()

            # ???
            if save_dir is not None:
                save_path = os.path.join(
                    save_dir,
                    f"AIwin_{safe_filename(feature_name)}_{w_i:02d}.png"
                )
                plt.savefig(save_path, dpi=200)
                print(f"[INFO] Saved plot: {save_path}")

            plt.show()
  
