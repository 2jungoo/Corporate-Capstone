import os, requests, datetime as dt, pandas as pd, sqlalchemy as sa, numpy as np
import time, urllib.parse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

KEY = (os.environ.get("KMA_KEY") or "").strip()
DB = os.environ.get("DB_URL")
NX = int(os.environ.get("KMA_NX", "60"))
NY = int(os.environ.get("KMA_NY", "127"))
URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
HDR = {"Accept": "application/json,*/*", "User-Agent": "Mozilla/5.0"}

try:
    requests.packages.urllib3.util.connection.HAS_IPV6 = False
except Exception:
    pass

def build_session():
    s = requests.Session()
    r = Retry(total=6, connect=3, read=3, backoff_factor=2, status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update(HDR)
    return s

def call_api(s, url, params, timeout=(10,90)):
    key_plain = params["serviceKey"]
    for k in [key_plain, urllib.parse.quote(key_plain, safe="")]:
        params["serviceKey"] = k
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code == 200 and r.text.strip().lower() != "forbidden":
                return r.json()
        except Exception:
            time.sleep(1)
    raise RuntimeError("Request failed")

def parse_items(items):
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    for c in ["fcstDate","fcstTime","baseDate","baseTime"]:
        if c in df.columns: df[c] = df[c].astype(str)
    df["fcst_dt"] = pd.to_datetime(df["fcstDate"] + df["fcstTime"], format="%Y%m%d%H%M", errors="coerce")
    df["base_dt"] = pd.to_datetime(df["baseDate"] + df["baseTime"], format="%Y%m%d%H%M", errors="coerce")
    keep = df[["base_dt","fcst_dt","category","fcstValue"]].dropna()
    w = keep.pivot_table(index=["base_dt","fcst_dt"], columns="category", values="fcstValue", aggfunc="last").reset_index()
    if 'TMP' in w.columns: w = w.rename(columns={'TMP':'T1H'})
    if 'PCP' in w.columns: w = w.rename(columns={'PCP':'RN1'})
    for c in ["T1H","REH","RN1","WSD","SKY","PTY"]:
        if c in w.columns:
            if c == "RN1":
                w[c] = w[c].replace({"강수없음":"0","1mm 미만":"0.1"}).astype(str).str.replace("mm","",regex=False)
            w[c] = pd.to_numeric(w[c], errors="coerce")
    w["nx"] = NX
    w["ny"] = NY
    return w

def fetch_for_base(s, base_dt_kst):
    bd = base_dt_kst.strftime("%Y%m%d")
    bt = base_dt_kst.strftime("%H%M")
    p = {"serviceKey": KEY, "dataType": "JSON", "numOfRows": 1000, "pageNo": 1, "base_date": bd, "base_time": bt, "nx": NX, "ny": NY}
    j = call_api(s, URL, p)
    it = ((j.get("response") or {}).get("body") or {}).get("items") or {}
    return parse_items(it.get("item") or [])

def bases_for_day(target_date_kst):
    prev = target_date_kst - dt.timedelta(days=1)
    lst = [dt.datetime(prev.year, prev.month, prev.day, 23, 0, 0)]
    for h in [2,5,8,11,14,17,20]:
        lst.append(dt.datetime(target_date_kst.year, target_date_kst.month, target_date_kst.day, h, 0, 0))
    return lst

def fetch_day_24h(s, target_date_kst):
    frames = []
    for b in bases_for_day(target_date_kst):
        w = fetch_for_base(s, b)
        if not w.empty:
            frames.append(w)
    if not frames:
        return pd.DataFrame()
    w = pd.concat(frames, ignore_index=True)
    start_dt = dt.datetime.combine(target_date_kst, dt.time(0,0,0))
    end_dt = start_dt + dt.timedelta(days=1)
    w = w[(w["fcst_dt"] >= start_dt) & (w["fcst_dt"] < end_dt)]
    if w.empty:
        return w
    w = w.sort_values(["fcst_dt","base_dt"])
    w = w.groupby("fcst_dt", as_index=False).apply(lambda g: g.ffill().iloc[-1]).reset_index(drop=True)
    return w

def upsert(w, e):
    if w.empty:
        return 0
    cols = ["nx","ny","base_dt","fcst_dt","T1H","REH","RN1","WSD","SKY","PTY"]
    have = [c for c in cols if c in w.columns]
    w = w[have].copy().replace({np.nan: None})
    if "RN1" in have and "PTY" in have:
        cond = w["PTY"].isin([1,2,3,4]) & (w["RN1"].isna() | (w["RN1"] == 0.0))
        w.loc[cond, "RN1"] = 1.0
    rows = w.to_dict("records")
    if not rows:
        return 0
    sql = (
        "INSERT INTO weather_ultra_fcst(" + ",".join(have) + ") VALUES(" + ",".join([f":{c}" for c in have]) + ") "
        "ON DUPLICATE KEY UPDATE " + ",".join([f"{c}=VALUES({c})" for c in have if c not in ("nx","ny","fcst_dt")])
    )
    with e.begin() as cx:
        cx.execute(sa.text(sql), rows)
    return len(rows)

def determine_target_date():
    s = (os.environ.get("TARGET_DATE") or "").strip()
    if s:
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            pass
    kst = dt.datetime.utcnow() + dt.timedelta(hours=9)
    return kst.date()

def delete_day(e, nx, ny, start_dt, end_dt):
    sql = "DELETE FROM weather_ultra_fcst WHERE nx=:nx AND ny=:ny AND fcst_dt>=:s AND fcst_dt<:e"
    with e.begin() as cx:
        r = cx.execute(sa.text(sql), {"nx": nx, "ny": ny, "s": start_dt, "e": end_dt})
        return r.rowcount if hasattr(r, "rowcount") else None

def main():
    if not KEY or not DB:
        raise SystemExit("set KMA_KEY and DB_URL")
    s = build_session()
    e = sa.create_engine(DB)
    tgt = determine_target_date()
    w = fetch_day_24h(s, tgt)
    start_dt = dt.datetime.combine(tgt, dt.time(0,0,0))
    end_dt = start_dt + dt.timedelta(days=1)
    deleted = delete_day(e, NX, NY, start_dt, end_dt)
    n = upsert(w, e)
    print({"deleted_rows": deleted, "upserted": n, "nx": NX, "ny": NY, "target_date": str(tgt), "total_hours": 0 if w is None else len(w)})

if __name__ == "__main__":
    main()
