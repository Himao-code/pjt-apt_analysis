# src/utils.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import re

# 수도권 분류
KOREA_CAPITAL = {"서울", "경기", "인천"}

# ---------------- 경로 유틸 ----------------
def ensure_project_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd, cwd.parent, cwd.parent.parent]:
        if (p / "src").exists() and ((p / "data").exists() or (p / "data.raw").exists()):
            return p
    return cwd

def get_data_dir(raw: bool = True) -> Path:
    root = ensure_project_root()
    if raw:
        return (root / "data" / "raw") if (root / "data" / "raw").exists() else (root / "data.raw")
    else:
        return root / "data" / "processed"

# ---------------- 내부 헬퍼 ----------------
def _read_csv_kr(path: Path) -> pd.DataFrame:
    """
    한국 부동산 CSV 안정 로더:
    - 인코딩 후보 순회
    - 안내문구를 건너뛰고 실제 헤더 라인 자동 탐지
    """
    encodings = ["ms949", "cp949", "euc-kr", "utf-8-sig", "utf-8", "latin1"]
    raw = path.read_bytes()
    header_idx, chosen_enc = None, None

    for enc in encodings:
        try:
            text = raw.decode(enc, errors="ignore")
            lines = text.splitlines()
            for i, line in enumerate(lines[:500]):
                if (("단지" in line and "거래금액" in line) or
                    ("전용면적" in line and "계약년월" in line)):
                    header_idx, chosen_enc = i, enc
                    break
            if chosen_enc:
                break
        except Exception:
            continue

    if chosen_enc is None:
        chosen_enc, header_idx = "ms949", 0

    df = pd.read_csv(path, encoding=chosen_enc, header=header_idx, engine="python")
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    return df

def _infer_city_from_name(name: str) -> str:
    m = re.search(r"(서울|인천|경기|부산|대구|대전|광주|울산|세종|제주|강원|충남|충북|전남|전북|경남|경북)", name)
    return m.group(1) if m else "미상"

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "본번": "bon", "부번": "bubun", "단지명": "complex",
        "전용면적(㎡)": "area_m2", "계약년월": "yyyymm", "계약일": "day",
        "거래금액(만원)": "price_manwon", "동": "dong", "층": "floor",
        "매수자": "buyer", "매도자": "seller", "건축년도": "built_year",
        "도로명": "road", "해제사유발생일": "cancel_date", "거래유형": "deal_type",
        "중개사소재지": "broker_region", "등기일자": "reg_date",
        "지역코드": "region_code", "시군구": "sigungu", "법정동": "beopjeongdong",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "price_manwon" in df.columns:
        df["price_manwon"] = (
            df["price_manwon"].astype(str).str.replace(",", "", regex=False).str.strip()
        )
        df["price_manwon"] = pd.to_numeric(df["price_manwon"], errors="coerce")

    if "area_m2" in df.columns:
        df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
        df["pyung"] = df["area_m2"] / 3.3

    if "yyyymm" in df.columns:
        df["yyyymm"] = df["yyyymm"].astype(str).str.slice(0, 6)
        if "day" in df.columns:
            day = df["day"].astype(str).str.zfill(2)
            df["date"] = pd.to_datetime(df["yyyymm"] + day, format="%Y%m%d", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["yyyymm"] + "01", format="%Y%m%d", errors="coerce")
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    if {"price_manwon", "area_m2"}.issubset(df.columns):
        df["price_per_m2_manwon"] = df["price_manwon"] / df["area_m2"]
    return df

# ---------------- 라벨링 ----------------
def apply_area_bucket(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 60, 85, np.inf]    # ㎡ 기준
    labels = ["소형", "중형", "대형"]
    df["size_bucket"] = pd.cut(df["area_m2"], bins=bins, labels=labels, right=False)
    return df

def apply_capital_label(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    df["region_group"] = np.where(df[city_col].isin(KOREA_CAPITAL), "수도권", "지방")
    return df

# ---------------- 로딩 ----------------
def load_local_raw(raw_dir: Path | str = None) -> pd.DataFrame:
    if raw_dir is None:
        raw_dir = get_data_dir(raw=True)
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV가 없습니다: {raw_dir.resolve()}")

    dfs = []
    for fp in files:
        df = _read_csv_kr(fp)
        df = _standardize_columns(df)
        df["city"] = _infer_city_from_name(fp.name)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    if "area_m2" in out.columns:
        out = apply_area_bucket(out)
    if "city" in out.columns:
        out = apply_capital_label(out, city_col="city")
    return out

# ---------------- 집계/분석 ----------------
def monthly_city_agg(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["region_group","city","month"], observed=True)
           .agg(price_per_m2_med=("price_per_m2_manwon","median"),
                price_per_m2_mean=("price_per_m2_manwon","mean"),
                total_price_med=("price_manwon","median"),
                volume=("price_manwon","size"))
           .reset_index())
    return g

def monthly_size_agg(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["city","size_bucket","month"], observed=True)
           .agg(price_per_m2_med=("price_per_m2_manwon","median"),
                volume=("price_manwon","size"))
           .reset_index())
    return g

def lagged_corr(x: pd.Series, y: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    res = []
    s1, s2 = x.sort_index(), y.sort_index()
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = s1.shift(lag).corr(s2)
        elif lag < 0:
            corr = s1.corr(s2.shift(-lag))
        else:
            corr = s1.corr(s2)
        res.append({"lag": lag, "corr": corr})
    return pd.DataFrame(res)

__all__ = [
    "ensure_project_root", "get_data_dir",
    "apply_area_bucket", "apply_capital_label",
    "load_local_raw", "monthly_city_agg", "monthly_size_agg", "lagged_corr",
]
