from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_INFO_DIR = ROOT / "db_info"
QC1_DIR = Path(os.getenv("QCAI_QC1_DIR", str(ROOT / "data" / "10_FLAG1")))
QC2_DIR = Path(os.getenv("QCAI_QC2_DIR", str(ROOT / "data" / "20_FLAG2")))


@dataclass(frozen=True)
class SourceSpec:
    table: str
    stn_id: str
    value_col: str
    time_col: str
    id_col: str
    qc1_tokens: List[str]
    qc2_tokens: List[str]


# Mapping rule provided by user
SOURCE_MAP: Dict[str, Dict[str, SourceSpec]] = {
    "SEA_LEVEL_SEA": {
        "Repr_Sea": SourceSpec(
            table="data_waterlevel",
            stn_id="B01",
            value_col="waterlevel",
            time_col="datetime",
            id_col="typ",
            qc1_tokens=["Repr_Sea_data_waterlevel_qc1_flags"],
            qc2_tokens=["Sihwa_Tide(해측)", "Sihwa_Basin(호측)"],
        )
    },
    "SEA_LEVEL_LAKE": {
        "Repr_Lake": SourceSpec(
            table="data_waterlevel",
            stn_id="B02",
            value_col="waterlevel",
            time_col="datetime",
            id_col="typ",
            qc1_tokens=["Repr_Lake_data_waterlevel_qc1_flags"],
            qc2_tokens=["Sihwa_Basin(호측)"],
        )
    },
    "SMB1_AIR": {
        "wind_speed": SourceSpec("data_air", "B01", "wind_spd", "obs_time", "stn_id", ["SMB1_data_air_qc1_flags"], ["SMB1_data_air_qc2_flags"]),
        "wind_max_speed": SourceSpec("data_air", "B01", "wind_max_spd", "obs_time", "stn_id", ["SMB1_data_air_qc1_flags"], ["SMB1_data_air_qc2_flags"]),
        "temp": SourceSpec("data_air", "B01", "air_temp", "obs_time", "stn_id", ["SMB1_data_air_qc1_flags"], ["SMB1_data_air_qc2_flags"]),
        "solar": SourceSpec("data_air", "B01", "insolation", "obs_time", "stn_id", ["SMB1_data_air_qc1_flags"], ["SMB1_data_air_qc2_flags"]),
        "wind_dir": SourceSpec("data_air", "B01", "wind_dir", "obs_time", "stn_id", ["SMB1_data_air_qc1_flags"], ["SMB1_data_air_qc2_flags"]),
        "press": SourceSpec("data_air", "B01", "air_press", "obs_time", "stn_id", ["SMB1_data_air_qc1_flags"], ["SMB1_data_air_qc2_flags"]),
    },
    "SMB2_AIR": {
        "wind_speed": SourceSpec("data_air", "B02", "wind_spd", "obs_time", "stn_id", ["SMB2_data_air_qc1_flags"], ["SMB2_data_air_qc2_flags"]),
        "wind_max_speed": SourceSpec("data_air", "B02", "wind_max_spd", "obs_time", "stn_id", ["SMB2_data_air_qc1_flags"], ["SMB2_data_air_qc2_flags"]),
        "temp": SourceSpec("data_air", "B02", "air_temp", "obs_time", "stn_id", ["SMB2_data_air_qc1_flags"], ["SMB2_data_air_qc2_flags"]),
        "solar": SourceSpec("data_air", "B02", "insolation", "obs_time", "stn_id", ["SMB2_data_air_qc1_flags"], ["SMB2_data_air_qc2_flags"]),
        "wind_dir": SourceSpec("data_air", "B02", "wind_dir", "obs_time", "stn_id", ["SMB2_data_air_qc1_flags"], ["SMB2_data_air_qc2_flags"]),
        "press": SourceSpec("data_air", "B02", "air_press", "obs_time", "stn_id", ["SMB2_data_air_qc1_flags"], ["SMB2_data_air_qc2_flags"]),
    },
    "SMB1_SURFACE": {
        "current_dir": SourceSpec("data_surface", "B01", "current_dir", "obs_time", "stn_id", ["SMB1_data_surface_qc1_flags"], ["SMB1_data_surface_qc2_flags"]),
        "current_speed": SourceSpec("data_surface", "B01", "current_spd", "obs_time", "stn_id", ["SMB1_data_surface_qc1_flags"], ["SMB1_data_surface_qc2_flags"]),
    },
    "SMB2_SURFACE": {
        "current_dir": SourceSpec("data_surface", "B02", "current_dir", "obs_time", "stn_id", ["SMB2_data_surface_qc1_flags"], ["SMB2_data_surface_qc2_flags"]),
        "current_speed": SourceSpec("data_surface", "B02", "current_spd", "obs_time", "stn_id", ["SMB2_data_surface_qc1_flags"], ["SMB2_data_surface_qc2_flags"]),
    },
    "SMB3_SURFACE": {
        "current_dir": SourceSpec("data_water", "B03", "current_dir", "obs_time", "stn_id", ["SMB3_data_surface_qc1_flags"], ["SMB3_data_surface_qc2_flags"]),
        "current_speed": SourceSpec("data_water", "B03", "current_spd", "obs_time", "stn_id", ["SMB3_data_surface_qc1_flags"], ["SMB3_data_surface_qc2_flags"]),
    },
    "SMB4_SURFACE": {
        "current_dir": SourceSpec("data_water", "B04", "current_dir", "obs_time", "stn_id", ["SMB4_data_surface_qc1_flags"], ["SMB4_data_surface_qc2_flags"]),
        "current_speed": SourceSpec("data_water", "B04", "current_spd", "obs_time", "stn_id", ["SMB4_data_surface_qc1_flags"], ["SMB4_data_surface_qc2_flags"]),
    },
    "SMB3_WATER": {
        "temp": SourceSpec("data_water", "B03", "water_temp", "obs_time", "stn_id", ["SMB3_data_water_qc1_flags"], ["SMB3_data_water_qc2_flags"]),
        "salinity": SourceSpec("data_water", "B03", "salinity", "obs_time", "stn_id", ["SMB3_data_water_qc1_flags"], ["SMB3_data_water_qc2_flags"]),
        "O2Per": SourceSpec("data_water", "B03", "o2_per", "obs_time", "stn_id", ["SMB3_data_water_qc1_flags"], ["SMB3_data_water_qc2_flags"]),
        "O2ppm": SourceSpec("data_water", "B03", "o2_ppm", "obs_time", "stn_id", ["SMB3_data_water_qc1_flags"], ["SMB3_data_water_qc2_flags"]),
        "pH": SourceSpec("data_water", "B03", "ph", "obs_time", "stn_id", ["SMB3_data_water_qc1_flags"], ["SMB3_data_water_qc2_flags"]),
        "chl-a": SourceSpec("data_water", "B03", "chl", "obs_time", "stn_id", ["SMB3_data_water_qc1_flags"], ["SMB3_data_water_qc2_flags"]),
        "turbidity": SourceSpec("data_water", "B03", "turbidity", "obs_time", "stn_id", ["SMB3_data_water_qc1_flags"], ["SMB3_data_water_qc2_flags"]),
    },
    "SMB4_WATER": {
        "temp": SourceSpec("data_water", "B04", "water_temp", "obs_time", "stn_id", ["SMB4_data_water_qc1_flags"], ["SMB4_data_water_qc2_flags"]),
        "salinity": SourceSpec("data_water", "B04", "salinity", "obs_time", "stn_id", ["SMB4_data_water_qc1_flags"], ["SMB4_data_water_qc2_flags"]),
        "O2Per": SourceSpec("data_water", "B04", "o2_per", "obs_time", "stn_id", ["SMB4_data_water_qc1_flags"], ["SMB4_data_water_qc2_flags"]),
        "O2ppm": SourceSpec("data_water", "B04", "o2_ppm", "obs_time", "stn_id", ["SMB4_data_water_qc1_flags"], ["SMB4_data_water_qc2_flags"]),
        "pH": SourceSpec("data_water", "B04", "ph", "obs_time", "stn_id", ["SMB4_data_water_qc1_flags"], ["SMB4_data_water_qc2_flags"]),
        "chl-a": SourceSpec("data_water", "B04", "chl", "obs_time", "stn_id", ["SMB4_data_water_qc1_flags"], ["SMB4_data_water_qc2_flags"]),
        "turbidity": SourceSpec("data_water", "B04", "turbidity", "obs_time", "stn_id", ["SMB4_data_water_qc1_flags"], ["SMB4_data_water_qc2_flags"]),
    },
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_db_manager():
    cfg = _load_module("db_config", DB_INFO_DIR / "0_db_config.py")
    sys.modules["db_config"] = cfg
    if str(DB_INFO_DIR) not in sys.path:
        sys.path.insert(0, str(DB_INFO_DIR))
    dbm = _load_module("database_manager_user", DB_INFO_DIR / "1_database_manager.py")
    return dbm.DatabaseManager, cfg.DB_CONFIG


def _parse_dt(dt_text: Optional[str], fallback: datetime) -> datetime:
    if not dt_text:
        return fallback
    return datetime.fromisoformat(dt_text)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    clean = []
    for c in df.columns:
        name = str(c).replace("\ufeff", "").strip()
        clean.append(name)
    df.columns = clean
    return df


def _pick_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = c.lower()
        if "obs_time" in cl or "datetime" in cl or "date" in cl or "time" in cl:
            return c
    return df.columns[0]


def _pick_qc_col(df: pd.DataFrame, value_col: str) -> Optional[str]:
    preferred = f"{value_col}_qc_result"
    if preferred in df.columns:
        return preferred
    cols = [c for c in df.columns if c.endswith("_qc_result")]
    return cols[0] if cols else None


def _collect_flag_files(base_dir: Path, tokens: List[str]) -> List[Path]:
    files = []
    for p in base_dir.glob("*.csv"):
        name = p.name
        if any(tok in name for tok in tokens):
            files.append(p)
    return sorted(files)


def _load_qc_flags(
    spec: SourceSpec,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    out = pd.DataFrame(columns=["date", "qc1", "qc2"])
    out["date"] = pd.to_datetime(out["date"])

    for stage, base_dir, tokens in [("qc1", QC1_DIR, spec.qc1_tokens), ("qc2", QC2_DIR, spec.qc2_tokens)]:
        frames = []
        for fp in _collect_flag_files(base_dir, tokens):
            try:
                d = _normalize_columns(_read_csv_safe(fp))
                tcol = _pick_time_col(d)
                qcol = _pick_qc_col(d, spec.value_col)
                if qcol is None:
                    continue
                sub = d[[tcol, qcol]].copy()
                sub.columns = ["date", stage]
                sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
                sub = sub.dropna(subset=["date"])
                frames.append(sub)
            except Exception:
                continue
        if frames:
            one = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"], keep="last")
            one = one[(one["date"] >= start_dt) & (one["date"] <= end_dt)]
            out = out.merge(one, on="date", how="outer", suffixes=("", "_new"))
            if f"{stage}_new" in out.columns:
                out[stage] = out[stage].combine_first(out[f"{stage}_new"])
                out = out.drop(columns=[f"{stage}_new"])

    out = out.sort_values("date").reset_index(drop=True)
    return out


class RealtimeDBLoader:
    def __init__(self):
        DatabaseManager, config = _load_db_manager()
        self.DatabaseManager = DatabaseManager
        self.config = config

    def fetch_feature(
        self,
        data_name: str,
        pair: str,
        start_dt: datetime,
        end_dt: datetime,
        include_flags: bool = True,
    ) -> pd.DataFrame:
        if data_name not in SOURCE_MAP or pair not in SOURCE_MAP[data_name]:
            raise ValueError(f"Unsupported data_name/pair: {data_name}/{pair}")
        spec = SOURCE_MAP[data_name][pair]

        start_s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_s = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        sql = (
            f"SELECT {spec.time_col} AS date, {spec.value_col} AS value "
            f"FROM {spec.table} "
            f"WHERE {spec.id_col} = '{spec.stn_id}' "
            f"AND {spec.time_col} BETWEEN '{start_s}' AND '{end_s}' "
            f"ORDER BY {spec.time_col}"
        )

        with self.DatabaseManager(self.config) as db:
            raw = db.execute_custom_query(sql)

        raw = _normalize_columns(raw)
        if raw.empty:
            frame = pd.DataFrame(columns=["date", "value"])
            frame["date"] = pd.to_datetime(frame["date"])
        else:
            frame = raw[["date", "value"]].copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if include_flags:
            flags = _load_qc_flags(spec, start_dt, end_dt)
            frame = frame.merge(flags, on="date", how="left")

        frame["data_name"] = data_name
        frame["pair"] = pair
        frame["table_name"] = spec.table
        frame["stn_id"] = spec.stn_id
        return frame

    def build_inference_input(
        self,
        data_name: str,
        pairs: Optional[List[str]],
        start_dt: datetime,
        end_dt: datetime,
        include_flags: bool = True,
    ) -> pd.DataFrame:
        if data_name not in SOURCE_MAP:
            raise ValueError(f"Unsupported data_name: {data_name}")
        use_pairs = pairs if pairs else list(SOURCE_MAP[data_name].keys())

        merged = None
        for pair in use_pairs:
            f = self.fetch_feature(data_name, pair, start_dt, end_dt, include_flags=include_flags)
            value_name = f"value_{pair}"
            cols = ["date", "value"]
            rename = {"value": value_name}
            if include_flags:
                # keep stage labels once per pair for debugging/monitoring
                rename["qc1"] = f"qc1_{pair}"
                rename["qc2"] = f"qc2_{pair}"
                cols.extend([c for c in ["qc1", "qc2"] if c in f.columns])
            f = f[cols].rename(columns=rename)
            merged = f if merged is None else merged.merge(f, on="date", how="outer")

        if merged is None:
            return pd.DataFrame(columns=["date"])
        merged = merged.sort_values("date").reset_index(drop=True)
        return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime DB loader for AI inference input")
    p.add_argument("--data-name", required=True, choices=sorted(SOURCE_MAP.keys()))
    p.add_argument("--pair", action="append", default=None, help="pair key. repeat for multi-pair")
    p.add_argument("--start-time", default=None, help="YYYY-mm-dd HH:MM:SS")
    p.add_argument("--end-time", default=None, help="YYYY-mm-dd HH:MM:SS")
    p.add_argument("--lookback-min", type=int, default=180, help="used when start-time is omitted")
    p.add_argument("--no-flags", action="store_true", help="skip QC1/QC2 merge")
    p.add_argument("--output-csv", default=None)
    p.add_argument("--print-head", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now()
    end_dt = _parse_dt(args.end_time, now)
    start_dt = _parse_dt(args.start_time, end_dt - timedelta(minutes=args.lookback_min))
    if start_dt >= end_dt:
        raise ValueError("start-time must be earlier than end-time")

    loader = RealtimeDBLoader()
    out = loader.build_inference_input(
        data_name=args.data_name,
        pairs=args.pair,
        start_dt=start_dt,
        end_dt=end_dt,
        include_flags=not args.no_flags,
    )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "data_name": args.data_name,
        "pairs": args.pair if args.pair else list(SOURCE_MAP[args.data_name].keys()),
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": int(len(out)),
        "columns": out.columns.tolist(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(out) > 0 and args.print_head > 0:
        print(out.head(args.print_head).to_string(index=False))


if __name__ == "__main__":
    main()
