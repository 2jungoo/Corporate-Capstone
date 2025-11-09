import os, re, json, glob
import numpy as np
import pandas as pd
from pathlib import Path

CFG = {
    "dataset": "./dummy_output_var/facilities_5m_all/facilities_5m_all_YYYYMMDD_HHMMSS.csv",
    "dataset_dir": "./dummy_output_var/facilities_5m_by_chamber",
    "ts_col": None,
    "power_cols": ["fan_power_w", "heating_value", "heating_power_kw"],
    "output_dir": "./energy",
    "daily_by_chamber_csv": "./energy/daily_energy_by_chamber.csv",
    "daily_total_csv": "./energy/daily_energy_total.csv",
    "daily_rate_csv": "./energy/daily_energy_rate_by_chamber.csv",
    "print_rows": 10
}

def read_csv(p):
    try:
        return pd.read_csv(p)
    except:
        return pd.read_csv(p, encoding="cp949")

def detect_ts_col(df):
    cands=[c for c in df.columns if any(t in str(c).lower() for t in ["time","ts","date","datetime"])]
    cands+=[c for c in df.columns if any(t in str(c) for t in ["시간","시각","일시"])]
    for c in cands+[df.columns[0]]:
        try:
            _=pd.to_datetime(df[c])
            return c
        except:
            pass
    raise ValueError("ts_col을 찾지 못했습니다")

def ensure_ts(df, ts_col):
    d=df.copy()
    d[ts_col]=pd.to_datetime(d[ts_col])
    return d.sort_values(ts_col).reset_index(drop=True)

def infer_chamber_from_name(path):
    s=os.path.basename(str(path))
    m=re.search(r"(?:ch(?:amber)?[_\-]?)?(\d+)", s, re.IGNORECASE)
    return int(m.group(1)) if m else np.nan

def load_source(CFG):
    paths=[]
    if CFG.get("dataset_dir") and os.path.isdir(CFG["dataset_dir"]):
        paths=sorted(glob.glob(os.path.join(CFG["dataset_dir"], "*.csv")))
    if not paths and CFG.get("dataset") and os.path.isfile(CFG["dataset"]):
        paths=[CFG["dataset"]]
    if not paths and CFG.get("dataset") and any(ch in CFG["dataset"] for ch in ["*", "?", "["]):
        paths=sorted(glob.glob(CFG["dataset"]))
    if not paths:
        raise ValueError("입력 CSV를 찾지 못했습니다")
    frames=[]
    for p in paths:
        df=read_csv(p)
        if "chamber" not in df.columns:
            ch=infer_chamber_from_name(p)
            df=df.assign(chamber=ch)
        frames.append(df)
    df=pd.concat(frames, ignore_index=True)
    ts_col=CFG["ts_col"] or detect_ts_col(df)
    df=ensure_ts(df, ts_col)
    return df, ts_col

def col_to_kw(series, name):
    n=name.lower()
    if "kw" in n:
        return series.astype(float)
    return (series.astype(float))/1000.0

def compute_daily(df, ts_col, power_cols):
    pcols=[c for c in power_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not pcols:
        raise ValueError("power_cols 중 사용 가능한 수치 컬럼이 없습니다")
    g=df.sort_values([ts_col,"chamber" if "chamber" in df.columns else ts_col]).copy()
    g["__date__"]=g[ts_col].dt.date
    if "chamber" in g.columns:
        parts=[]
        for ch, gg in g.groupby("chamber", dropna=False):
            gg=gg.sort_values(ts_col).copy()
            gg["__dt_h__"]=gg[ts_col].diff().dt.total_seconds().fillna(0)/3600.0
            for c in pcols:
                gg[f"__e_{c}__"]=col_to_kw(gg[c].ffill(), c)*gg["__dt_h__"]
            gg["energy_total_kwh"]=gg[[f"__e_{c}__" for c in pcols]].sum(axis=1)
            parts.append(gg)
        D=pd.concat(parts, axis=0).reset_index(drop=True)
        agg_cols={"energy_total_kwh":"sum"}
        for c in pcols:
            agg_cols[f"__e_{c}__"]="sum"
        daily=D.groupby(["__date__","chamber"], dropna=False).agg(agg_cols).reset_index()
        daily=daily.rename(columns={"__date__":"date"})
        for c in pcols:
            daily=daily.rename(columns={f"__e_{c}__":f"{c}_kwh"})
        daily["date"]=pd.to_datetime(daily["date"])
        daily=daily.sort_values(["date","chamber"]).reset_index(drop=True)
        daily_tot=(daily.groupby(["date"], dropna=False)[["energy_total_kwh"]+[f"{c}_kwh" for c in pcols]].sum().reset_index().sort_values("date"))
        return daily, daily_tot, pcols
    else:
        gg=g.sort_values(ts_col).copy()
        gg["__dt_h__"]=gg[ts_col].diff().dt.total_seconds().fillna(0)/3600.0
        for c in pcols:
            gg[f"__e_{c}__"]=col_to_kw(gg[c].ffill(), c)*gg["__dt_h__"]
        gg["energy_total_kwh"]=gg[[f"__e_{c}__" for c in pcols]].sum(axis=1)
        daily=(gg.groupby([gg[ts_col].dt.date], dropna=False)
               .agg({**{f"__e_{c}__":"sum" for c in pcols}, "energy_total_kwh":"sum"})
               .reset_index())
        daily=daily.rename(columns={"index":"date"})
        for c in pcols:
            daily=daily.rename(columns={f"__e_{c}__":f"{c}_kwh"})
        daily["date"]=pd.to_datetime(daily["date"])
        daily=daily.sort_values("date").reset_index(drop=True)
        return daily.assign(chamber=np.nan), daily.copy(), pcols

def compute_rate(daily_by_ch, daily_tot, pcols):
    tot=daily_tot[["date","energy_total_kwh"]].rename(columns={"energy_total_kwh":"__tot__"})
    d=daily_by_ch.merge(tot, on="date", how="left")
    d["energy_rate_pct"]=np.where(d["__tot__"]>0, d["energy_total_kwh"]/d["__tot__"]*100.0, np.nan)
    d=d.drop(columns="__tot__")
    d=d.sort_values(["date","chamber"]).reset_index(drop=True)
    return d

def main():
    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
    df, ts_col = load_source(CFG)
    daily_by_ch, daily_tot, used_pcols = compute_daily(df, ts_col, CFG["power_cols"])
    Path(CFG["daily_by_chamber_csv"]).parent.mkdir(parents=True, exist_ok=True)
    daily_by_ch.to_csv(CFG["daily_by_chamber_csv"], index=False)
    daily_tot.to_csv(CFG["daily_total_csv"], index=False)
    rate_by_ch=compute_rate(daily_by_ch, daily_tot, used_pcols)
    rate_by_ch.to_csv(CFG["daily_rate_csv"], index=False)
    head_rows=min(CFG["print_rows"], len(daily_tot))
    info={
        "inputs": {"dataset": CFG.get("dataset"), "dataset_dir": CFG.get("dataset_dir")},
        "ts_col": ts_col,
        "power_cols_used": used_pcols,
        "output_by_chamber_csv": CFG["daily_by_chamber_csv"],
        "output_total_csv": CFG["daily_total_csv"],
        "output_rate_csv": CFG["daily_rate_csv"],
        "rows_daily_by_chamber": int(len(daily_by_ch)),
        "rows_daily_total": int(len(daily_tot)),
        "rows_daily_rate": int(len(rate_by_ch))
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("── daily_total (tail) ──")
    print(daily_tot.tail(head_rows).to_string(index=False))
    print("── daily_rate_by_chamber % (tail) ──")
    pv=(rate_by_ch.pivot(index="date", columns="chamber", values="energy_rate_pct").sort_index())
    print(pv.tail(head_rows).to_string())

if __name__=="__main__":
    main()
