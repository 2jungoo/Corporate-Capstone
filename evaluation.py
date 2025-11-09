# evaluation_tuned.py
import os, json, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from xgboost import XGBRegressor

CFG = {
    "dataset": "./dummy_output_var/facilities_5m_all/facilities_5m_all_20251109_025417.csv",
    "bundle_meta": "./model/bundle_meta.json",
    "model_path": "./model/xgb_unified.json",
    "output_csv": "./model/alerts_pred.csv",
    "band_mode": "mad",
    "k": 1.5,
    "min_std": 1e-6,
    "band_window_periods": 96,
    "max_pred_path": 4,
    "float_ndigits": 3,
    "w_path": 0.6,
    "w_trend": 0.4
}

def read_csv(p):
    try: return pd.read_csv(p)
    except: return pd.read_csv(p, encoding="cp949")

def ensure_ts(df,ts_col):
    d=df.copy(); d[ts_col]=pd.to_datetime(d[ts_col]); return d.sort_values(ts_col).reset_index(drop=True)

def resample(df,ts_col,rule):
    return df.set_index(ts_col).sort_index().resample(rule).ffill().bfill().reset_index()

def time_feats(s):
    t=pd.to_datetime(s); hod=t.dt.hour+t.dt.minute/60.0; dow=t.dt.dayofweek
    return pd.DataFrame({"hod_sin":np.sin(2*np.pi*hod/24.0),"hod_cos":np.cos(2*np.pi*hod/24.0),"dow_sin":np.sin(2*np.pi*dow/7.0),"dow_cos":np.cos(2*np.pi*dow/7.0)})

def build_self_feats(series,lags,wins):
    X={}
    for i in range(1,lags+1): X[f"self_lag_{i}"]=series.shift(i)
    for w in wins:
        X[f"self_sma_{w}"]=series.rolling(w).mean()
        X[f"self_std_{w}"]=series.rolling(w).std()
    return pd.DataFrame(X)

def build_driver_feats(df, drivers, lags_cross, ts_col):
    F=[df[[ts_col]].copy()]
    for d in drivers:
        if d not in df.columns or not pd.api.types.is_numeric_dtype(df[d]): continue
        for i in range(1,lags_cross+1):
            F[-1][f"{d}_lag_{i}"]=df[d].shift(i)
    return pd.concat(F,axis=1)

def baseline_band(a, mode="sigma", k=3.0, min_std=1e-6):
    x=np.asarray(a, dtype=float)
    if mode=="mad":
        med=float(np.nanmedian(x))
        mad=float(np.nanmedian(np.abs(x-med)))
        s=max(1.4826*mad, min_std)
        return med-k*s, med+k*s, med, s
    mu=float(np.nanmean(x)); s=float(max(np.nanstd(x), min_std))
    return mu-k*s, mu+k*s, mu, s

def fnum(v,nd=3):
    try: return f"{float(v):.{nd}f}"
    except: return str(v)

def arr_last(df,ts_col,need):
    return df.sort_values(ts_col).tail(need).copy()

def build_X_for_var(df, meta, var):
    ts_col=meta["ts_col"]; rule=meta["rule"]; lags_self=meta["lags_self"]; wins_self=meta["wins_self"]
    drivers=meta["drivers"]; lags_cross=meta["lags_cross"]; feature_cols=meta["feature_cols"]
    s=df[var]; SF=build_self_feats(s,lags_self,wins_self)
    DR=build_driver_feats(df,drivers,lags_cross,ts_col).drop(columns=[ts_col])
    TF=time_feats(df[ts_col])
    base=pd.concat([df[[ts_col]], SF, DR, TF], axis=1).dropna().reset_index(drop=True)
    if base.empty: return None, None
    last=base.iloc[[-1]].copy()
    rows=[]; metas=[]
    for h in meta["horizons"]:
        Z=last.copy(); Z["var_name"]=var; Z["var_id"]=meta["var_map"][var]; Z["h"]=h
        Z["var_id"]=Z["var_id"].astype("category"); Z["h"]=Z["h"].astype("category")
        Z=pd.get_dummies(Z, columns=["var_id","h"], drop_first=False, dtype=np.float32)
        for c in feature_cols:
            if c not in Z.columns: Z[c]=np.float32(0.0)
        Z=Z[feature_cols].astype(np.float32)
        rows.append(Z); metas.append({"h":h})
    X=pd.concat(rows,axis=0).reset_index(drop=True)
    return X, metas

def main():
    meta=json.load(open(CFG["bundle_meta"],"r",encoding="utf-8"))
    ts_col=meta["ts_col"]; rule=meta["rule"]; lags_self=meta["lags_self"]; wins_self=meta["wins_self"]
    horizons=meta["horizons"]; value_cols=meta["value_cols"]; stats=meta["var_target_stats"]
    model=XGBRegressor(); model.load_model(CFG["model_path"])
    df0=read_csv(CFG["dataset"]); df0=ensure_ts(df0, ts_col); df=resample(df0, ts_col, rule)
    need=max(lags_self, max(wins_self), max(horizons))+1
    out=[]
    for var in tqdm(value_cols, desc="Predict"):
        if var not in df.columns or not pd.api.types.is_numeric_dtype(df[var]): continue
        hist=arr_last(df[[ts_col,var]], ts_col, need)
        if len(hist)<need: continue
        win=min(len(hist), int(CFG["band_window_periods"]))
        base_vals=hist[var].values[-win:]
        lb,ub,mu,sd=baseline_band(base_vals, CFG["band_mode"], CFG["k"], CFG["min_std"])
        last_ts=hist[ts_col].iloc[-1]
        Xdf, metas=build_X_for_var(df, meta, var)
        if Xdf is None: continue
        yhat_z=model.predict(Xdf.to_numpy(np.float32))
        den=lambda z: z*stats[var]["std"]+stats[var]["mean"]
        preds=[]
        for i,m in enumerate(metas):
            t_pred=last_ts+pd.tseries.frequencies.to_offset(rule)*int(m["h"])
            preds.append({"h":int(m["h"]), "ts":pd.Timestamp(t_pred).isoformat(), "yhat": float(den(yhat_z[i]))})
        current=float(hist[var].values[-1])
        zs=[]
        outs=[]
        for p in preds:
            z=abs((p["yhat"]-mu)/max(sd, CFG["min_std"]))
            zs.append(z)
            outs.append(1.0 if (p["yhat"]>ub or p["yhat"]<lb) else 0.0)
        ratio=float(np.mean(outs)) if len(outs)>0 else 0.0
        excess=[max(0.0, z-CFG["k"])/max(CFG["k"],1e-6) for z in zs]
        mean_excess=float(np.mean(excess)) if len(excess)>0 else 0.0
        risk_path=float(0.5*ratio+0.5*mean_excess)
        dy_last=abs(preds[-1]["yhat"]-current) if len(preds)>0 else 0.0
        risk_trend=float(min(1.0, dy_last/(CFG["k"]*max(sd, CFG["min_std"]))))
        risk=float(np.clip(CFG["w_path"]*risk_path+CFG["w_trend"]*risk_trend, 0.0, 1.0))
        eta=None; h_=None; direction=None
        for p in preds:
            if p["yhat"]>ub: eta=p["ts"]; h_=p["h"]; direction="up"; break
            if p["yhat"]<lb: eta=p["ts"]; h_=p["h"]; direction="down"; break
        if eta is None and len(preds)>0:
            direction="up" if preds[-1]["yhat"]>current else ("down" if preds[-1]["yhat"]<current else None)
        w=hist[var].values[-max(4, int(len(hist)*0.25)):]
        slope=float(pd.Series(w).diff().mean())
        out.append({
            "variable":var,
            "eta":eta,
            "h":None if h_ is None else int(h_),
            "direction":direction,
            "baseline_mean":float(mu),
            "baseline_std":float(sd),
            "current":float(current),
            "band_lower":float(lb),
            "band_upper":float(ub),
            "trend_slope":slope,
            "risk":risk,
            "pred_path":preds
        })
    nd=int(CFG["float_ndigits"])
    print("─"*80); print(f"예측 변수 개수: {len(out)}  (rule={rule}, lags={lags_self}, horizons={horizons})"); print("─"*80)
    for a in out:
        var=a["variable"]; eta=a["eta"] if a["eta"] else "예측 없음"; h="-" if a["h"] is None else str(a["h"])
        d="↑" if a["direction"]=="up" else ("↓" if a["direction"]=="down" else "·")
        cur=fnum(a["current"],nd); mu=fnum(a["baseline_mean"],nd); sd=fnum(a["baseline_std"],nd)
        lb=fnum(a["band_lower"],nd); ub=fnum(a["band_upper"],nd); slope=fnum(a["trend_slope"],nd)
        rk=float(a["risk"]); rp=a["pred_path"][:CFG["max_pred_path"]]
        dy="-"
        if len(rp)>0:
            try: dy0=float(rp[-1]["yhat"])-float(a["current"]); dy=("+" if dy0>=0 else "")+fnum(dy0,nd)
            except: pass
        bar="█"*int(round(rk*20))+"·"*int(20-int(round(rk*20)))
        print(f"[{var}] {d}  위험도 {int(round(rk*100))}%  {bar}")
        print(f"  ETA: {eta}  (h={h})  방향: {a['direction'] if a['direction'] else '-'}")
        print(f"  현재값: {cur}  기준중앙/평균: {mu}  스케일: {sd}")
        print(f"  밴드: [{lb}, {ub}]  단기추세기울기: {slope}  장기예상변화: {dy}")
        if len(rp)>0:
            print("  예측경로:")
            for e in rp: print(f"    - h={e['h']}→{e['ts']} : {fnum(e['yhat'], nd)}")
        print("─"*80)
    rows=[{
        "variable":a["variable"],
        "eta":a["eta"],
        "h":a["h"],
        "direction":a["direction"],
        "baseline_mean":a["baseline_mean"],
        "baseline_std":a["baseline_std"],
        "current":a["current"],
        "band_lower":a["band_lower"],
        "band_upper":a["band_upper"],
        "trend_slope":a["trend_slope"],
        "risk":a["risk"],
        "pred_path_json":json.dumps(a["pred_path"],ensure_ascii=False)
    } for a in out]
    out_csv=Path(CFG["output_csv"]); out_csv.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

if __name__=="__main__":
    main()
