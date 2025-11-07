import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import glob

rng = np.random.default_rng()

CONFIG = {
    "profiles": {
        "fac_5m": {
            "glob": "./csv/facilities_5m_chamber*.csv",
            "encodings": ["utf-8", "cp949", "euc-kr", None],
            "freq": "5min",
            "timestamp_cols": {"date": "date", "time": "time"},
            "rename_map": {}
        },
        "fac_1h": {
            "glob": "./csv/facilities_1h_chamber*.csv",
            "encodings": ["utf-8", "cp949", "euc-kr", None],
            "freq": "1h",
            "timestamp_cols": {"date": "date", "time": "time"},
            "rename_map": {}
        },
        "pig": {
            "glob": "./csv/pig_chamber*.csv",
            "encodings": ["cp949", "utf-8", "euc-kr", None],
            "freq": "1d",
            "timestamp_cols": {"date": "수집날짜", "time": "수집시간"},
            "rename_map": {
                "분당호흡횟수": "resp_per_min",
                "직장온도(℃)": "rectal_temp_c",
                "피부온도(등)": "skin_temp_back",
                "피부온도(목)": "skin_temp_neck",
                "피부온도(머리)": "skin_temp_head",
                "돼지 체중(kg)": "weight_kg"
            }
        }
    },
    "rows_per_profile": 1000,
    "duration_days": 30,
    "start_iso": "2023-08-03 00:00:00",
    "seed": 123,
    "base_dir": "./dummy_output_var",
    "sigma_scale": 1.0,
    "anom_rate": 0.0025,
    "pig_weight_daily_gain": 0.55,
    "pig_unhealthy_gain_scale": 0.35,
    "pig_unhealthy_temp_add": 1.0,
    "pig_unhealthy_resp_add": 20.0,
    "pig_start_weight_range": [20.0, 70.0],
    "chambers": [1, 2, 3, 4],
    "pigs_per_chamber": 10,
    "unhealthy_pig_id": 10,
    "fac1h_update_every_steps": 12,
    "max_delta_per_5m": {
        "T_0.5": 0.8,
        "T_1.5": 0.8,
        "inside_temp": 0.8,
        "outside_temp": 1.2,
        "RH_0.5": 4.0,
        "RH_1.5": 4.0,
        "inside_humidity": 4.0,
        "outside_humidity": 5.0
    },
    "slaughter_weight_kg": 118.0,
    "pig_class_bounds_kg": {
        "piglet": [0.0, 30.0],
        "pig": [30.0, 60.0],
        "grown_pig": [60.0, 118.0]
    },
    "pig_adg_kg_per_day": {
        "piglet": 0.32,
        "pig": 0.75,
        "grown_pig": 0.90
    },
    "pig_finisher_taper_start_kg": 100.0,
    "pig_finisher_taper_end_kg": 118.0,
    "slaughter_weight_kg": 118.0,
    "pig_class_bounds_kg": {
        "piglet": [0.0, 30.0],
        "pig": [30.0, 60.0],
        "grown_pig": [60.0, 118.0]
    },
    "pig_adfi_curve_kg_per_day": {
        "piglet_start_w": 13.0,
        "piglet_start_adfi": 0.5,
        "piglet_end_w": 30.0,
        "piglet_end_adfi": 1.2,
        "pig_start_w": 30.0,
        "pig_start_adfi": 1.6,
        "pig_end_w": 60.0,
        "pig_end_adfi": 2.0,
        "grown_start_w": 60.0,
        "grown_start_adfi": 2.3,
        "grown_plateau_w": 110.0,
        "grown_plateau_adfi": 2.5,
        "grown_end_w": 118.0,
        "grown_end_adfi": 2.6
    },
    "pig_unhealthy_feed_scale": 0.9,
    "pig_feed_range_kg_per_day": {
    "piglet": [0.65, 0.85],
    "pig": [1.70, 1.95],
    "grown_pig": [2.35, 2.55]
    },
    "feed_rand_std_ratio": 0.08,
    "feed_smooth_alpha": 0.6,
    "pig_feed_range_kg_per_day": {
    "piglet": [0.50, 1.00],
    "pig": [1.55, 2.05],
    "grown_pig": [2.25, 2.65]
    },
    "feed_rand_std_ratio": 0.18,
    "feed_max_daily_change_kg": 0.30,
    "feed_rand_std_ratio": 0.14,
    "feed_blend_margin_kg": 8.0,
    "feed_transition_glide": 0.40,
    "feed_max_daily_change_kg": 0.25,
    "seed": None,
    "pig_init_mix": {"piglet": 0.35, "pig": 0.45, "grown_pig": 0.20},
    "pig_init_weight_range_by_class": {
        "piglet": [8.0, 25.0],
        "pig": [30.0, 55.0],
        "grown_pig": [60.0, 85.0]
    },
    "pig_init_mix": {"piglet": 0.50, "pig": 0.35, "grown_pig": 0.15},
    "min_piglet_per_chamber": 3,
    "seed": None,
    "unhealthy_pig_id": 1,
    "pig_unhealthy_gain_scale": 0.7,
    "pig_unhealthy_feed_scale": 0.85,
    "pig_unhealthy_temp_add": 0.8,
    "pig_unhealthy_resp_add": 6,
    "sick_mode": "daily",
    "sick_prob_per_day": 0.05,
    "disease_weights": {"fever":0.55,"enteritis":0.25,"cold":0.10,"cough":0.10},
    "disease_global_scale": {"gain":0.75,"feed":0.80,"temp":1.40,"resp":1.30},
    "disease_class_sensitivity": {
    "piglet": {"gain":0.85,"feed":0.85,"temp":1.20,"resp":1.20},
    "pig": {"gain":1.00,"feed":1.00,"temp":1.00,"resp":1.00},
    "grown_pig": {"gain":0.95,"feed":0.95,"temp":0.90,"resp":1.00}
    },
}

EXCLUDE_COLS = {
    "chamber","챔버","챔버번호","pig_id","돼지번호",
    "room","section","unit_id","device_id","profile"
}

FALLBACK = {
    "fac_5m": {
        "evaporation": (60.54,20.0,0,120),
        "T_0.5": (25.59,0.7,20,30),
        "RH_0.5": (31.82,7.0,10,60),
        "CO2_0.5": (821.27,150,300,2000),
        "NH3_0.5": (11.29,1.0,0,40),
        "T_1.5": (26.08,0.7,20,30),
        "RH_1.5": (19.82,6.0,5,60),
        "CO2_1.5": (852.545,160,300,2000),
        "NH3_1.5": (5.16,0.8,0,40),
        "fan_rpm": (900,150,0,1800),
        "fan_power_w": (120,30,0,300)
    },
    "fac_1h": {
        "ventilation": (2.08,0.45,0,10),
        "heating_value": (883.76,95,0,1500),
        "set_temp": (25.0,0.05,25,25),
        "outside_temp": (18.0,6.0,-10,35),
        "inside_temp": (25.6,0.9,15,35),
        "outside_humidity": (62.0,10.0,10,100),
        "inside_humidity": (45.0,12.0,10,100)
    },
    "pig": {
        "resp_per_min": (63,6,20,120),
        "rectal_temp_c": (38.0,0.6,35,41.5),
        "skin_temp_back": (34.2,1.0,25,41),
        "skin_temp_neck": (34.0,1.0,25,41),
        "skin_temp_head": (34.3,0.9,25,41),
        "weight_kg": (60.3,20,5,200)
    }
}

DATE_FORMATS = ["%Y-%m-%d","%Y/%m/%d","%Y.%m.%d","%Y%m%d","%m/%d/%Y","%d/%m/%Y"]
TIME_FORMATS = ["%H:%M:%S","%H:%M","%H%M%S","%H%M","%I:%M:%S %p","%I:%M %p"]

def _pig_class_of(w, cfg):
    b = cfg["pig_class_bounds_kg"]
    if w < b["piglet"][1]:
        return "piglet"
    elif w < b["pig"][1]:
        return "pig"
    else:
        return "grown_pig"

def _mid(a, b):
    return 0.5 * (a + b)

def _blend(a, b, r):
    if r < 0.0: r = 0.0
    if r > 1.0: r = 1.0
    return a * (1.0 - r) + b * r

def _class_range(cls, cfg):
    lo, hi = cfg["pig_feed_range_kg_per_day"][cls]
    return float(lo), float(hi)

def _blend_range(w, cfg):
    m = float(cfg["feed_blend_margin_kg"])
    p_end = cfg["pig_class_bounds_kg"]["piglet"][1]
    g_start = cfg["pig_class_bounds_kg"]["pig"][1]
    if p_end - m <= w <= p_end + m:
        lo0, hi0 = _class_range("piglet", cfg)
        lo1, hi1 = _class_range("pig", cfg)
        r = (w - (p_end - m)) / (2.0 * m)
        return _blend(lo0, lo1, r), _blend(hi0, hi1, r)
    if g_start - m <= w <= g_start + m:
        lo0, hi0 = _class_range("pig", cfg)
        lo1, hi1 = _class_range("grown_pig", cfg)
        r = (w - (g_start - m)) / (2.0 * m)
        return _blend(lo0, lo1, r), _blend(hi0, hi1, r)
    cls = _pig_class_of(w, cfg)
    return _class_range(cls, cfg)

def _feedstuff_draw(w, cls, prev_feed, prev_cls, cfg):
    lo, hi = _blend_range(w, cfg)
    mu = _mid(lo, hi)
    sigma = (hi - lo) * float(cfg["feed_rand_std_ratio"])
    v = rng.normal(mu, sigma)
    if v < lo: v = lo
    if v > hi: v = hi
    if prev_feed is not None and prev_cls is not None and cls != prev_cls:
        g = float(cfg["feed_transition_glide"])
        v = prev_feed + g * (v - prev_feed)
    if prev_feed is not None:
        dmax = float(cfg["feed_max_daily_change_kg"])
        dv = v - prev_feed
        if dv > dmax: v = prev_feed + dmax
        if dv < -dmax: v = prev_feed - dmax
    return v

def _pig_class_of(w, cfg):
    b = cfg["pig_class_bounds_kg"]
    if w < b["piglet"][1]:
        return "piglet"
    elif w < b["pig"][1]:
        return "pig"
    else:
        return "grown_pig"

def _feedstuff_random(cls, cfg, prev=None):
    lo, hi = cfg["pig_feed_range_kg_per_day"][cls]
    mu = 0.5 * (lo + hi)
    sigma = (hi - lo) * float(cfg["feed_rand_std_ratio"])
    v = rng.normal(mu, sigma)
    if v < lo: v = lo
    if v > hi: v = hi
    a = float(cfg.get("feed_smooth_alpha", 0.0))
    if prev is not None and 0.0 < a < 1.0:
        v = a * prev + (1.0 - a) * v
        if v < lo: v = lo
        if v > hi: v = hi
    return v

def _pig_class_of(w, cfg):
    b = cfg["pig_class_bounds_kg"]
    if w < b["piglet"][1]:
        return "piglet"
    elif w < b["pig"][1]:
        return "pig"
    else:
        return "grown_pig"

def _interp(x, x0, y0, x1, y1):
    if x1 == x0:
        return y0
    r = (x - x0) / (x1 - x0)
    r = 0.0 if r < 0.0 else 1.0 if r > 1.0 else r
    return y0 + r * (y1 - y0)

def _adfi_for_w(w, cfg):
    c = cfg["pig_adfi_curve_kg_per_day"]
    if w < c["piglet_start_w"]:
        return max(0.4, _interp(w, 5.0, 0.4, c["piglet_start_w"], c["piglet_start_adfi"]))
    if w < c["piglet_end_w"]:
        return _interp(w, c["piglet_start_w"], c["piglet_start_adfi"], c["piglet_end_w"], c["piglet_end_adfi"])
    if w < c["pig_end_w"]:
        return _interp(w, c["pig_start_w"], c["pig_start_adfi"], c["pig_end_w"], c["pig_end_adfi"])
    if w < c["grown_plateau_w"]:
        return _interp(w, c["grown_start_w"], c["grown_start_adfi"], c["grown_plateau_w"], c["grown_plateau_adfi"])
    if w < c["grown_end_w"]:
        return _interp(w, c["grown_plateau_w"], c["grown_plateau_adfi"], c["grown_end_w"], c["grown_end_adfi"])
    return c["grown_end_adfi"]

def _pig_class_of(w, cfg):
    b = cfg["pig_class_bounds_kg"]
    if w < b["piglet"][1]:
        return "piglet"
    elif w < b["pig"][1]:
        return "pig"
    else:
        return "grown_pig"

def _adg_for(w, cls, cfg):
    adg = float(cfg["pig_adg_kg_per_day"][cls])
    if cls == "grown_pig" and w >= float(cfg["pig_finisher_taper_start_kg"]):
        s = float(cfg["pig_finisher_taper_start_kg"])
        e = float(cfg["pig_finisher_taper_end_kg"])
        span = max(e - s, 1e-6)
        r = min(max((w - s) / span, 0.0), 1.0)
        adg = adg * (1.0 - 0.4 * r)
    return adg

def ts_fmt(ts):
    return ts.strftime("%Y-%m-%d  %H:%M:%S")

def clamp(v,a,b):
    return np.minimum(np.maximum(v,a),b)

def _clean_numeric_frame(df):
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace(r"[^\d\.\-\+eE]", "", regex=True)
            df[c] = pd.to_numeric(s, errors="coerce")
    return df

def read_one_csv(path, encodings, rename_map):
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
            if rename_map:
                df = df.rename(columns=rename_map)
            df = _clean_numeric_frame(df)
            return df
        except:
            continue
    return None

def _best_dt_format(date_s: pd.Series, time_s: pd.Series):
    date_s = date_s.astype(str); time_s = time_s.astype(str)
    combo = date_s + " " + time_s
    best_fmt = None; best_ok = -1
    for dfmt in DATE_FORMATS:
        for tfmt in TIME_FORMATS:
            fmt = f"{dfmt} {tfmt}"
            try:
                parsed = pd.to_datetime(combo, format=fmt, errors="coerce")
                ok = parsed.notna().sum()
                if ok > best_ok:
                    best_ok = ok; best_fmt = fmt
            except:
                pass
    return best_fmt

def load_many(cfg):
    files = sorted(glob.glob(cfg["glob"]))
    frames = []
    for f in files:
        df = read_one_csv(f, cfg["encodings"], cfg.get("rename_map", {}))
        if df is None: continue
        drop_cols = [c for c in df.columns if str(c) in EXCLUDE_COLS]
        if drop_cols: df = df.drop(columns=drop_cols, errors="ignore")
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    dcol = cfg["timestamp_cols"].get("date")
    tcol = cfg["timestamp_cols"].get("time")
    if dcol in df.columns and tcol in df.columns:
        fmt = _best_dt_format(df[dcol], df[tcol])
        if fmt:
            ts = pd.to_datetime(df[dcol].astype(str)+" "+df[tcol].astype(str), format=fmt, errors="coerce")
        else:
            ds = df[dcol].astype(str).str.replace(r"[.\/]", "-", regex=True)
            ts = pd.to_datetime(ds+" "+df[tcol].astype(str), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    else:
        cand = None
        for c in df.columns:
            if "timestamp" in str(c).lower():
                cand = c; break
        if cand is None:
            return None
        best = None; best_ok = -1
        for dfmt in DATE_FORMATS:
            for tfmt in TIME_FORMATS:
                fmt = f"{dfmt} {tfmt}"
                try:
                    parsed = pd.to_datetime(df[cand].astype(str), format=fmt, errors="coerce")
                    ok = parsed.notna().sum()
                    if ok>best_ok:
                        best_ok=ok; best=fmt
                except:
                    pass
        if best:
            ts = pd.to_datetime(df[cand].astype(str), format=best, errors="coerce")
        else:
            s = df[cand].astype(str).str.replace(r"[.\/]", "-", regex=True)
            ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
    return df

def _impute_numeric(num: pd.DataFrame):
    num = num.sort_index().interpolate(limit_direction="both")
    med = num.median(numeric_only=True); num = num.fillna(med)
    return num

def fit_from_df_or_fallback(df, prof_key):
    if df is None or df.empty:
        base = FALLBACK[prof_key]
        cols = [c for c in base.keys() if c not in EXCLUDE_COLS]
        mu = np.array([base[c][0] for c in cols], dtype=float)
        std = np.array([max(base[c][1],1e-6) for c in cols], dtype=float)
        cov = np.diag(std**2)
        phi = np.full(len(cols), 0.6, dtype=float)
        season = pd.DataFrame(0.0, index=range(24), columns=cols)
        bounds = {c:(base[c][2], base[c][3]) for c in cols}
        step_q90 = np.array(std)
        return {"cols":cols,"mu":mu,"cov":cov,"phi":phi,"season":season,"bounds":bounds,"step_q90":step_q90}
    num = df.select_dtypes(include=["number"]).copy()
    if "timestamp" in num.columns: num = num.drop(columns=["timestamp"])
    rm = [c for c in num.columns if c in EXCLUDE_COLS]
    if rm: num = num.drop(columns=rm, errors="ignore")
    if num.shape[1]==0: return fit_from_df_or_fallback(None, prof_key)
    num = _impute_numeric(num)
    cols = num.columns.tolist()
    mu = num.mean().values
    h = df["timestamp"].dt.hour
    Xc = num - mu
    season = Xc.groupby(h).mean().reindex(range(24)).fillna(0.0)
    Xcs = Xc.values - season.loc[h].values
    if Xcs.shape[0] < 3:
        phi = np.full(len(cols),0.5); resid = Xcs
    else:
        Xlag = Xcs[:-1]; Xcur = Xcs[1:]; phi=[]
        for j in range(len(cols)):
            a = Xlag[:,j]; b = Xcur[:,j]; den = (a*a).sum()
            phij = (a*b).sum()/den if den>1e-12 else 0.5
            phi.append(float(np.clip(phij,0.0,0.99)))
        phi = np.array(phi); resid = Xcur - Xlag*phi
    if resid.shape[0] > 1:
        cov = np.cov(resid.T)
    else:
        v = np.maximum(np.var(resid, axis=0), 1e-6); cov = np.diag(v)
    dX = num.diff().dropna()
    dXa = dX.abs(); step_q90 = dXa.quantile(0.90).reindex(cols).fillna(0.0).values
    step_q90 = np.maximum(step_q90, 1e-6)
    q01 = num.quantile(0.01); q99 = num.quantile(0.99)
    bounds = {}
    for c in cols:
        lo = float(q01[c]); hi = float(q99[c])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo>=hi:
            fb = FALLBACK.get(prof_key, {}).get(c, None)
            if fb: lo, hi = fb[2], fb[3]
            else:
                m = float(mu[cols.index(c)]); d = 3.0; lo, hi = m-d, m+d
        bounds[c]=(lo,hi)
    return {"cols":cols,"mu":mu,"cov":cov,"phi":phi,"season":season,"bounds":bounds,"step_q90":step_q90}

def chol_psd(mat):
    try:
        return np.linalg.cholesky(mat)
    except:
        d = np.diag(np.maximum(np.diag(mat),1e-8)); return np.linalg.cholesky(d)

def next_vector(state, prev_x, hour, sigma_scale):
    mu = state["mu"]; phi = state["phi"]; s = state["season"].iloc[hour].values
    L = chol_psd(state["cov"])
    if prev_x is None:
        z = rng.standard_normal(len(state["cols"])); step = (L @ z) * sigma_scale
        return mu + s + step
    base_prev = mu + s
    xcs_prev = prev_x - base_prev
    pred = mu + s + phi * xcs_prev
    z = rng.standard_normal(len(state["cols"])); step = (L @ z) * sigma_scale
    return pred + step

def limit_step_named(prev, cur, step_q90, cols, overrides):
    if prev is None: return cur
    delta = cur - prev
    caps = step_q90.copy()
    for i, c in enumerate(cols):
        if c in overrides:
            caps[i] = min(caps[i], float(overrides[c]))
    delta = np.clip(delta, -caps, caps)
    return prev + delta

def inject_anomaly(profile, cols, row, ctx):
    if profile == "pig":
        return
    if rng.random() < ctx["anom_rate"] and len(cols)>0:
        t = rng.choice(["spike","drop","noisy"])
        target_cols = cols
        if t=="spike":
            k = rng.choice(target_cols); row[k]=float(row[k]*rng.uniform(3,6))
        elif t=="drop":
            k = rng.choice(target_cols); row[k]=float(row[k]*rng.uniform(0.05,0.3))
        elif t=="noisy":
            k = rng.choice(target_cols); row[k]=float(row[k]+rng.normal(0,10))
    if profile=="fac_5m":
        nh3 = [c for c in cols if "NH3" in c]
        if rng.random() < ctx["anom_rate"]/2 and nh3:
            for k in nh3: row[k]=float(rng.uniform(30,40))
            if "fan_rpm" in cols: row["fan_rpm"]=0.0
            if "fan_power_w" in cols: row["fan_power_w"]=0.0

def post_delta_cap(prev_row, row, overrides, bounds):
    if prev_row is None: return row
    out = dict(row)
    for k, v in row.items():
        if k in overrides and k in prev_row:
            d = float(v) - float(prev_row[k])
            cap = float(overrides[k])
            if d > cap: v = float(prev_row[k]) + cap
            elif d < -cap: v = float(prev_row[k]) - cap
            lo, hi = bounds.get(k, (-np.inf, np.inf))
            v = float(np.clip(v, lo, hi))
            out[k] = v
    return out

def steps_for_5min(days, rows_hint):
    if days is not None:
        return int(np.ceil(pd.Timedelta(days=days) / pd.to_timedelta("5min")))
    return int(rows_hint)

def gen_facilities_5m_all(cfg):
    df5 = load_many(cfg["profiles"]["fac_5m"])
    st5 = fit_from_df_or_fallback(df5, "fac_5m")
    df1 = load_many(cfg["profiles"]["fac_1h"])
    st1 = fit_from_df_or_fallback(df1, "fac_1h")
    b5 = st5["bounds"]; b1 = st1["bounds"]
    q5 = st5["step_q90"]; q1 = st1["step_q90"]
    cols5 = st5["cols"]; cols1 = st1["cols"]
    start = cfg["start_iso"] or datetime.now(timezone.utc).isoformat()
    ts = pd.Timestamp(start)
    steps = steps_for_5min(cfg.get("duration_days"), cfg["rows_per_profile"])
    prev5_vec = None
    prev5_row = None
    prev1_row = None
    hold = 0
    out = []
    sp = int(max(cfg["fac1h_update_every_steps"],1))
    x1_start = None
    x1_end = None
    while len(out) < steps:
        hour = ts.hour
        x5_raw = next_vector(st5, prev5_vec, hour, cfg["sigma_scale"])
        x5_l = limit_step_named(prev5_vec if prev5_vec is not None else x5_raw, x5_raw, q5, cols5, cfg["max_delta_per_5m"])
        lo5 = np.array([b5[c][0] for c in cols5]); hi5 = np.array([b5[c][1] for c in cols5])
        x5 = clamp(x5_l, lo5, hi5)
        row5 = {c: float(x5[j]) for j,c in enumerate(cols5)}
        inject_anomaly("fac_5m", cols5, row5, {"anom_rate": cfg["anom_rate"]})
        row5 = post_delta_cap(prev5_row, row5, cfg["max_delta_per_5m"], b5)
        prev5_vec = np.array([row5[c] for c in cols5], dtype=float)
        prev5_row = dict(row5)
        if hold % sp == 0 or x1_start is None or x1_end is None:
            x1_prev = x1_end if x1_end is not None else None
            x1_base = next_vector(st1, x1_prev, hour, cfg["sigma_scale"])
            x1_base = limit_step_named(x1_prev if x1_prev is not None else x1_base, x1_base, q1, cols1, cfg["max_delta_per_5m"])
            lo1 = np.array([b1[c][0] for c in cols1]); hi1 = np.array([b1[c][1] for c in cols1])
            x1_start = clamp(x1_base, lo1, hi1)
            nx_raw = next_vector(st1, x1_start, hour, cfg["sigma_scale"])
            nx_lim = limit_step_named(x1_start, nx_raw, q1, cols1, cfg["max_delta_per_5m"])
            x1_end = clamp(nx_lim, lo1, hi1)
            hold = 0
        alpha = float(hold + 1) / float(sp)
        x1 = x1_start + (x1_end - x1_start) * alpha
        row1 = {c: float(x1[j]) for j,c in enumerate(cols1)}
        inject_anomaly("fac_1h", cols1, row1, {"anom_rate": cfg["anom_rate"]})
        row1 = post_delta_cap(prev1_row, row1, cfg["max_delta_per_5m"], b1)
        rec = {"ts": ts_fmt(ts.to_pydatetime())}
        rec.update(row5); rec.update(row1)
        out.append(rec)
        prev1_row = dict(row1)
        hold += 1
        ts = ts + pd.tseries.frequencies.to_offset("5min")
    out_dir = Path(cfg["base_dir"]) / "facilities_5m_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"facilities_5m_all_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    path = out_dir / fname
    pd.DataFrame(out).to_csv(path, index=False)
    return path

def gen_pig_daily(cfg):
    P = cfg["profiles"]["pig"]
    df = load_many(P)
    st = fit_from_df_or_fallback(df, "pig")
    bounds = st["bounds"]; cols = st["cols"]
    start = cfg["start_iso"] or datetime.now(timezone.utc).isoformat()
    start_ts = pd.Timestamp(start).normalize()
    days = int(cfg.get("duration_days", 30))
    out = []
    for chamber in cfg["chambers"]:
        n_pigs = int(cfg["pigs_per_chamber"])
        mode = str(cfg.get("sick_mode", "one_per_chamber_per_day"))
        unhealthy_id = int(cfg.get("unhealthy_pig_id", 1))
        diseases = cfg.get("diseases", {
            "cold": {"gain_scale":0.85,"feed_scale":0.90,"temp_add":0.6,"resp_add":4},
            "enteritis": {"gain_scale":0.70,"feed_scale":0.75,"temp_add":0.5,"resp_add":2},
            "fever": {"gain_scale":0.60,"feed_scale":0.80,"temp_add":1.2,"resp_add":8},
            "cough": {"gain_scale":0.90,"feed_scale":0.95,"temp_add":0.3,"resp_add":6}
        })
        weights_cfg = cfg.get("disease_weights", {"cold":0.4,"enteritis":0.25,"fever":0.2,"cough":0.15})
        dn = [k for k in diseases.keys() if k in weights_cfg]
        wp = np.array([weights_cfg[k] for k in dn], dtype=float)
        wp = wp / wp.sum() if wp.sum() > 0 else np.ones(len(dn))/max(len(dn),1)
        gs = cfg.get("disease_global_scale", {"gain":1.0,"feed":1.0,"temp":1.0,"resp":1.0})
        cs = cfg.get("disease_class_sensitivity", {})
        daily_sick_ids = None
        daily_diseases = None
        if mode == "one_per_chamber_per_day":
            daily_sick_ids = [int(rng.integers(1, n_pigs + 1)) for _ in range(days)]
            daily_diseases = [dn[int(rng.choice(len(dn), p=wp))] for _ in range(days)]
        use_init_mix = ("pig_init_mix" in cfg) and ("pig_init_weight_range_by_class" in cfg)
        init_classes = []
        init_weights = []
        if use_init_mix:
            mix = cfg["pig_init_mix"]
            p_piglet = float(mix["piglet"]); p_pig = float(mix["pig"]); p_grown = float(mix["grown_pig"])
            for _ in range(n_pigs):
                u = rng.random()
                if u < p_piglet:
                    init_classes.append("piglet")
                elif u < p_piglet + p_pig:
                    init_classes.append("pig")
                else:
                    init_classes.append("grown_pig")
            need = max(0, int(cfg.get("min_piglet_per_chamber", 0)) - sum(1 for c in init_classes if c == "piglet"))
            if need > 0:
                for i in range(n_pigs):
                    if init_classes[i] != "piglet":
                        init_classes[i] = "piglet"
                        need -= 1
                        if need == 0:
                            break
            w_ranges = cfg["pig_init_weight_range_by_class"]
            for cls0 in init_classes:
                w_lo, w_hi = w_ranges[cls0]
                init_weights.append(float(rng.uniform(w_lo, w_hi)))
        for idx, pig_id in enumerate(range(1, n_pigs + 1)):
            prev_vec = None
            prev_feed = None
            prev_cls = None
            w_idx = cols.index("weight_kg") if "weight_kg" in cols else None
            if w_idx is None:
                continue
            if use_init_mix:
                w = init_weights[idx]
            else:
                w = float(rng.uniform(cfg["pig_start_weight_range"][0], cfg["pig_start_weight_range"][1]))
            for d in range(days):
                ts = start_ts + pd.Timedelta(days=d)
                x_raw = next_vector(st, prev_vec, 0, cfg["sigma_scale"])
                lo = np.array([bounds[c][0] for c in cols]); hi = np.array([bounds[c][1] for c in cols])
                x = clamp(x_raw, lo, hi)
                cls = _pig_class_of(w, cfg)
                adg = _adg_for(w, cls, cfg)
                is_sick = False
                disease_name = "healthy"
                if mode == "fixed":
                    is_sick = (pig_id == unhealthy_id)
                    if is_sick:
                        disease_name = cfg.get("fixed_disease", dn[int(rng.choice(len(dn), p=wp))])
                elif mode == "daily":
                    if rng.random() < float(cfg.get("sick_prob_per_day", 0.05)):
                        is_sick = True
                        disease_name = dn[int(rng.choice(len(dn), p=wp))]
                elif mode == "one_per_chamber_per_day":
                    is_sick = (pig_id == daily_sick_ids[d])
                    if is_sick:
                        disease_name = daily_diseases[d]
                if is_sick:
                    prm = diseases.get(disease_name, {"gain_scale":1.0,"feed_scale":1.0,"temp_add":0.0,"resp_add":0.0})
                    cm = cs.get(cls, {"gain":1.0,"feed":1.0,"temp":1.0,"resp":1.0})
                    gscale = float(prm.get("gain_scale",1.0)) * float(gs.get("gain",1.0)) * float(cm.get("gain",1.0))
                    adg *= gscale
                w = min(w + adg, float(cfg["slaughter_weight_kg"]))
                x[w_idx] = float(np.clip(w, bounds["weight_kg"][0], bounds["weight_kg"][1]))
                feedstuff_kg = _feedstuff_draw(w, cls, prev_feed, prev_cls, cfg)
                if is_sick:
                    prm = diseases.get(disease_name, {"gain_scale":1.0,"feed_scale":1.0,"temp_add":0.0,"resp_add":0.0})
                    cm = cs.get(cls, {"gain":1.0,"feed":1.0,"temp":1.0,"resp":1.0})
                    fscale = float(prm.get("feed_scale",1.0)) * float(gs.get("feed",1.0)) * float(cm.get("feed",1.0))
                    feedstuff_kg *= fscale
                row = {c: float(x[j]) for j, c in enumerate(cols)}
                if is_sick:
                    prm = diseases.get(disease_name, {"gain_scale":1.0,"feed_scale":1.0,"temp_add":0.0,"resp_add":0.0})
                    cm = cs.get(cls, {"gain":1.0,"feed":1.0,"temp":1.0,"resp":1.0})
                    tadd = float(prm.get("temp_add",0.0)) * float(gs.get("temp",1.0)) * float(cm.get("temp",1.0))
                    radd = float(prm.get("resp_add",0.0)) * float(gs.get("resp",1.0)) * float(cm.get("resp",1.0))
                    if "rectal_temp_c" in row:
                        row["rectal_temp_c"] = float(np.clip(row["rectal_temp_c"] + tadd, bounds["rectal_temp_c"][0], bounds["rectal_temp_c"][1]))
                    for k in ["skin_temp_back","skin_temp_neck","skin_temp_head"]:
                        if k in row:
                            row[k] = float(np.clip(row[k] + tadd*0.8, bounds[k][0], bounds[k][1]))
                    if "resp_per_min" in row:
                        row["resp_per_min"] = float(np.clip(row["resp_per_min"] + radd, bounds["resp_per_min"][0], bounds["resp_per_min"][1]))
                prev_vec = np.array([row[c] for c in cols], dtype=float)
                prev_feed = feedstuff_kg
                prev_cls = cls
                rec = {
                    "chamber": chamber,
                    "pig_id": pig_id,
                    "pig_class": cls,
                    "sick": 1 if is_sick else 0,
                    "disease": disease_name,
                    "feedstuff_kg": round(float(feedstuff_kg), 3),
                    "ts": ts_fmt(ts.to_pydatetime())
                }
                rec.update(row)
                out.append(rec)
    out_df = pd.DataFrame(out)
    first_cols = ["chamber","pig_id","pig_class","sick","disease","feedstuff_kg","ts"]
    other_cols = [c for c in out_df.columns if c not in first_cols]
    out_df = out_df[first_cols + other_cols]
    out_dir = Path(cfg["base_dir"]) / "pig_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"pig_daily_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    path = out_dir / fname
    out_df.to_csv(path, index=False)
    return path

def main():
    if CONFIG["seed"] is not None:
        global rng
        rng = np.random.default_rng(CONFIG["seed"])
    p1 = gen_facilities_5m_all(CONFIG)
    p2 = gen_pig_daily(CONFIG)
    print(str(p1)); print(str(p2))

if __name__ == "__main__":
    print("Hello")
    main()
