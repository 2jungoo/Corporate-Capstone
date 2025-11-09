# model.py  (통합 단일 모델 학습/저장: XGBoost 3.x Booster + 조기종료 콜백)
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb

CFG = {
    "dataset": "./dummy_output_var/facilities_5m_all/facilities_train_dataset.csv",
    "ts_col": None,
    "exclude_cols": ["set_temp","chamber","pig_id","pig_class","sick","disease"],
    "rule": "15min",
    "lags_self": 32,
    "wins_self": [4,8,16,32],
    "horizons": [1,2,4,8,12],
    "drivers": ["outside_temp","outside_humidity","inside_temp","inside_humidity","ventilation","fan_rpm","fan_power_w","heating_value"],
    "lags_cross": 4,
    "test_size": 0.2,
    "random_state": 42,
    "xgb_params": {
        "n_estimators": 1200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
    },
    "early_stopping_rounds": 100,
    "models_dir": "./model",
    "unified_model_path": "./model/xgb_unified.json",
    "bundle_meta": "./model/bundle_meta.json",
    "metrics_csv": "./model/train_metrics.csv"
}

def read_csv(p):
    try: return pd.read_csv(p)
    except: return pd.read_csv(p, encoding="cp949")

def detect_ts_col(df):
    cands=[c for c in df.columns if any(t in c.lower() for t in ["time","ts","date"])]
    cands+= [c for c in df.columns if any(t in c for t in ["시간","시각"])]
    for c in cands+[df.columns[0]]:
        try:
            _=pd.to_datetime(df[c]); return c
        except: pass
    raise ValueError("시계열 컬럼을 찾지 못했습니다. CFG['ts_col']을 지정하세요.")

def ensure_ts(df,ts_col):
    d=df.copy(); d[ts_col]=pd.to_datetime(d[ts_col]); return d.sort_values(ts_col).reset_index(drop=True)

def resample(df,ts_col,rule):
    return df.set_index(ts_col).sort_index().resample(rule).ffill().bfill().reset_index()

def time_feats(s):
    t=pd.to_datetime(s); hod=t.dt.hour+t.dt.minute/60.0; dow=t.dt.dayofweek
    return pd.DataFrame({
        "hod_sin":np.sin(2*np.pi*hod/24.0),
        "hod_cos":np.cos(2*np.pi*hod/24.0),
        "dow_sin":np.sin(2*np.pi*dow/7.0),
        "dow_cos":np.cos(2*np.pi*dow/7.0),
    })

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

def detect_value_cols(df, exclude):
    ex=set(exclude or [])
    cols=[c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: raise ValueError("수치 컬럼을 찾지 못했습니다. CFG['value_cols']를 지정하세요.")
    return cols

def gpu_capable():
    try:
        X=np.random.rand(64,8).astype("float32"); y=np.random.rand(64).astype("float32")
        dtrain=xgb.DMatrix(X, label=y)
        params={"objective":"reg:squarederror","tree_method":"hist","device":"cuda:0","max_depth":1,"eta":0.3}
        xgb.train(params, dtrain, num_boost_round=1)
        return True
    except Exception:
        return False

def chron_split_idx(n, test_size):
    n_tr=int(round(n*(1.0-test_size)))
    return max(1, min(n-1, n_tr))

def to_long_dataset(df, ts_col, value_cols, horizons, lags_self, wins_self, drivers, lags_cross):
    DR=build_driver_feats(df, drivers, lags_cross, ts_col)
    TF=time_feats(df[ts_col])
    frames=[]
    var_map={v:i for i,v in enumerate(value_cols)}
    for v in tqdm(value_cols, desc="Build"):
        s=df[v]
        SF=build_self_feats(s,lags_self,wins_self)
        base=pd.concat([df[[ts_col]], SF, DR.drop(columns=[ts_col]), TF], axis=1)
        for h in horizons:
            y=s.shift(-h).rename("y")
            Z=pd.concat([base,y],axis=1)
            Z["var_name"]=v; Z["var_id"]=var_map[v]; Z["h"]=h
            frames.append(Z)
    D=pd.concat(frames,axis=0).dropna().reset_index(drop=True)
    return D, var_map, list(DR.drop(columns=[ts_col]).columns), list(TF.columns)

def one_hot_keep_h(D):
    D=D.copy()
    D["_h_raw"]=D["h"]
    D["var_id"]=D["var_id"].astype("category")
    D["h"]=D["h"].astype("category")
    D=pd.get_dummies(D, columns=["var_id","h"], drop_first=False, dtype=np.float32)
    D.rename(columns={"_h_raw":"h"}, inplace=True)
    return D

def main():
    Path(CFG["models_dir"]).mkdir(parents=True,exist_ok=True)
    df0=read_csv(CFG["dataset"])
    ts_col=CFG["ts_col"] or detect_ts_col(df0)
    df0=ensure_ts(df0, ts_col)
    df=resample(df0, ts_col, CFG["rule"])
    value_cols=detect_value_cols(df, CFG["exclude_cols"])
    drivers=[d for d in CFG["drivers"] if d in df.columns and d not in CFG["exclude_cols"]]

    D,var_map,driver_feat_cols,time_feat_cols=to_long_dataset(
        df,ts_col,value_cols,CFG["horizons"],CFG["lags_self"],CFG["wins_self"],drivers,CFG["lags_cross"]
    )
    Denc=one_hot_keep_h(D)
    feature_cols=[c for c in Denc.columns if c not in [ts_col,"y","var_name","h"]]
    Denc[feature_cols]=Denc[feature_cols].astype(np.float32)
    Denc=Denc.sort_values(ts_col).reset_index(drop=True)

    n=len(df)
    n_tr_idx=chron_split_idx(n, CFG["test_size"])
    split_ts=df[ts_col].iloc[n_tr_idx]

    vnames=Denc["var_name"].values
    y_raw=Denc["y"].astype(float).values
    var_stats={}
    for v in value_cols:
        s_tr=df.loc[df[ts_col]<=split_ts, v].astype(float)
        mu=float(s_tr.mean()); sd=float(s_tr.std(ddof=0));
        if sd<=0 or np.isnan(sd): sd=1e-6
        var_stats[v]={"mean":mu,"std":sd}
    mu_arr=np.vectorize(lambda v: var_stats[v]["mean"])(vnames)
    sd_arr=np.vectorize(lambda v: var_stats[v]["std"])(vnames)
    y_z=(y_raw-mu_arr)/sd_arr

    tr_mask=(Denc[ts_col]<=split_ts).values
    Xtr=Denc.loc[tr_mask, feature_cols].to_numpy(np.float32)
    ytr=y_z[tr_mask].astype(np.float32)
    Xte=Denc.loc[~tr_mask, feature_cols].to_numpy(np.float32)
    yte=y_z[~tr_mask].astype(np.float32)

    use_gpu=gpu_capable()
    print(f"[INFO] Backend: {'GPU(cuda:0)' if use_gpu else 'CPU'}  | samples: train={len(Xtr)}, test={len(Xte)}, features={len(feature_cols)}")

    params={
        "objective":"reg:squarederror",
        "eval_metric":"rmse",
        "tree_method":"hist",
        "device":"cuda:0" if use_gpu else "cpu",
        "max_depth": int(CFG["xgb_params"].get("max_depth", 8)),
        "eta": float(CFG["xgb_params"].get("learning_rate", 0.05)),
        "subsample": float(CFG["xgb_params"].get("subsample", 0.8)),
        "colsample_bytree": float(CFG["xgb_params"].get("colsample_bytree", 0.8)),
        "lambda": float(CFG["xgb_params"].get("reg_lambda", 1.0)),
        "seed": int(CFG["random_state"])
    }
    num_boost_round=int(CFG["xgb_params"].get("n_estimators", 1200))

    dtrain=xgb.DMatrix(Xtr, label=ytr, feature_names=feature_cols)
    dvalid=xgb.DMatrix(Xte, label=yte, feature_names=feature_cols)

    callbacks=[]
    if int(CFG.get("early_stopping_rounds", 0))>0:
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=int(CFG["early_stopping_rounds"]),
                metric_name="rmse",
                data_name="validation",
                save_best=True
            )
        )

    booster=xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain,"train"),(dvalid,"validation")],
        callbacks=callbacks
    )

    yhat_z=booster.predict(dvalid)
    te_meta=Denc.loc[~tr_mask,[ts_col,"var_name","h"]].copy()
    denorm=lambda v,z: z*var_stats[v]["std"]+var_stats[v]["mean"]
    # 실제 y 복원
    yte_real=np.vectorize(denorm)(te_meta["var_name"].values, yte)
    yhat_real=np.vectorize(denorm)(te_meta["var_name"].values, yhat_z)
    te_meta["y"]=yte_real.astype(float)
    te_meta["yhat"]=yhat_real.astype(float)

    grp=(te_meta.groupby(["var_name","h"], as_index=False, group_keys=False)
         .apply(lambda g: pd.Series({"mae": float(np.mean(np.abs(g["y"]-g["yhat"]))), "n": len(g)}))
         .reset_index(drop=True))
    Path(CFG["metrics_csv"]).parent.mkdir(parents=True,exist_ok=True)
    grp.to_csv(CFG["metrics_csv"], index=False)

    Path(CFG["unified_model_path"]).parent.mkdir(parents=True,exist_ok=True)
    booster.save_model(str(CFG["unified_model_path"]))

    meta={
        "model_path": CFG["unified_model_path"],
        "device_used": "cuda:0" if use_gpu else "cpu",
        "ts_col": ts_col,
        "rule": CFG["rule"],
        "lags_self": CFG["lags_self"],
        "wins_self": CFG["wins_self"],
        "horizons": CFG["horizons"],
        "value_cols": value_cols,
        "exclude_cols": CFG["exclude_cols"],
        "drivers": drivers,
        "lags_cross": CFG["lags_cross"],
        "feature_cols": feature_cols,
        "driver_feat_cols": driver_feat_cols,
        "time_feat_cols": time_feat_cols,
        "split_ts": str(split_ts),
        "var_map": {k:int(v) for k,v in var_map.items()},
        "var_target_stats": var_stats,
        "xgb_params": params,
        "num_boost_round": num_boost_round
    }
    with open(CFG["bundle_meta"],"w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)

    print(json.dumps({
        "unified_model_path": CFG["unified_model_path"],
        "bundle_meta": CFG["bundle_meta"],
        "metrics_csv": CFG["metrics_csv"],
        "n_variables": len(value_cols)
    }, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
