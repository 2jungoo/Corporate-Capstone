import json
import numpy as np
import pandas as pd
from pathlib import Path

CFG = {
    "alerts_csv": "./dummy_output_var/facilities_5m_all/alerts.csv",
    "sort_by": "risk",
    "descending": True,
    "top_n": None,
    "risk_threshold": 0.0,
    "max_pred_path": 4,
    "float_ndigits": 3
}

def read_csv_smart(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, encoding="cp949")

def fnum(v, nd=3):
    if v is None:
        return "None"
    try:
        if isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v)):
            return "NaN"
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

def risk_bar(r, n=20):
    try:
        x = float(r)
    except Exception:
        x = 0.0
    if x < 0: x = 0.0
    if x > 1: x = 1.0
    k = int(round(x * n))
    return "█"*k + "·"*(n-k)

def arrow(direction):
    if direction == "up":
        return "↑"
    if direction == "down":
        return "↓"
    return "·"

def parse_pred_path(s):
    if s is None or (isinstance(s,float) and np.isnan(s)):
        return []
    if isinstance(s, list):
        return s
    try:
        return json.loads(s)
    except Exception:
        return []

def load_alerts(path):
    df = read_csv_smart(path)
    if "pred_path_json" in df.columns:
        df["pred_path"] = df["pred_path_json"].apply(parse_pred_path)
    elif "pred_path" in df.columns:
        df["pred_path"] = df["pred_path"].apply(parse_pred_path)
    else:
        df["pred_path"] = [[] for _ in range(len(df))]
    cols = ["variable","eta","h","direction","baseline_mean","baseline_std","current","band_lower","band_upper","trend_slope","risk","pred_path"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    if CFG["sort_by"] in df.columns:
        df = df.sort_values(CFG["sort_by"], ascending=not CFG["descending"])
    if CFG["risk_threshold"] > 0:
        df = df[df["risk"] >= CFG["risk_threshold"]]
    if CFG["top_n"]:
        df = df.head(int(CFG["top_n"]))
    return df.reset_index(drop=True)

def print_line():
    print("─"*80)

def print_alert_row(row):
    nd = int(CFG["float_ndigits"])
    var = str(row["variable"])
    eta = row["eta"] if pd.notna(row["eta"]) else "예측 없음"
    h = str(int(row["h"])) if pd.notna(row["h"]) else "-"
    d = arrow(row["direction"])
    cur = fnum(row["current"], nd)
    mu = fnum(row["baseline_mean"], nd)
    sd = fnum(row["baseline_std"], nd)
    lb = fnum(row["band_lower"], nd)
    ub = fnum(row["band_upper"], nd)
    slope = fnum(row["trend_slope"], nd)
    rk = float(row["risk"]) if pd.notna(row["risk"]) else 0.0
    bar = risk_bar(rk)
    rp = row["pred_path"][: CFG["max_pred_path"]]
    dy = None
    if len(rp) > 0 and "yhat" in rp[0]:
        try:
            dy = float(rp[0]["yhat"]) - float(row["current"])
        except Exception:
            dy = None
    delta_str = f"+{fnum(dy, nd)}" if isinstance(dy, (float,int)) and not np.isnan(dy) else "-"
    header = f"[{var}] {d}  위험도 {int(round(rk*100))}%  {bar}"
    print(header)
    print(f"  ETA: {eta}  (h={h})  방향: {row['direction'] if pd.notna(row['direction']) else '-'}")
    print(f"  현재값: {cur}  기준평균: {mu}  표준편차: {sd}")
    print(f"  밴드: [{lb}, {ub}]  단기추세기울기: {slope}  예상변화량(가까운 수평선): {delta_str}")
    if len(rp) > 0:
        s = []
        for e in rp:
            hh = e.get("h","-")
            tt = e.get("ts","-")
            yh = fnum(e.get("yhat", None), nd)
            s.append(f"h={hh}→{tt} : {yh}")
        print("  예측경로:")
        for chunk in s:
            print(f"    - {chunk}")

def main():
    p = CFG["alerts_csv"]
    if not p or not Path(p).exists():
        raise FileNotFoundError("CFG['alerts_csv'] 경로를 확인하세요.")
    df = load_alerts(p)
    print_line()
    print(f"알림 개수: {len(df)}  (정렬: {CFG['sort_by']}, 임계치: {CFG['risk_threshold']}, 최대경로표시: {CFG['max_pred_path']})")
    print_line()
    for _, row in df.iterrows():
        print_alert_row(row)
        print_line()

if __name__ == "__main__":
    main()
