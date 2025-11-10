import argparse, time, json, math, random, numpy as np, pandas as pd
from datetime import datetime, timedelta

class PigHealthRuleEngine:
    def __init__(self, mode="test", seed=42, freq_min=60, minutes=720, chambers=4, pigs_per_chamber=6, out_signals="signals.csv", out_alerts="alerts.csv"):
        self.mode=mode
        self.rng=np.random.default_rng(seed)
        self.freq=timedelta(minutes=freq_min)
        self.minutes=minutes
        self.chambers=chambers
        self.ppc=pigs_per_chamber
        self.out_signals=out_signals
        self.out_alerts=out_alerts
        self.rules=[(0,30,38.7,39.9),(30,70,38.6,39.8),(70,1e9,38.5,39.7)]

    def _normal_range(self,w):
        for lo,hi,tmin,tmax in self.rules:
            if lo<=w<hi: return tmin,tmax
        return 38.6,39.8

    def _severity(self,temp,tmin,tmax):
        if math.isnan(temp): return "MISSING", None
        if tmin<=temp<=tmax: return "OK", 0.0
        d=min(abs(temp-tmin),abs(temp-tmax))
        if d<0.3: return "CAUTION", d
        if d<0.8: return "WARN", d
        return "CRITICAL", d

    def _eval_df(self,df):
        rows=[]
        for _,r in df.iterrows():
            tmin,tmax=self._normal_range(r["weight_kg"])
            sev,score=self._severity(r["rectal_temp"],tmin,tmax)
            if sev!="OK":
                rows.append({
                    "timestamp":r["timestamp"],
                    "chamber":int(r["chamber"]),
                    "pig_id":int(r["pig_id"]),
                    "weight_kg":float(r["weight_kg"]),
                    "rectal_temp":(None if pd.isna(r["rectal_temp"]) else float(r["rectal_temp"])),
                    "tmin":tmin,
                    "tmax":tmax,
                    "severity":sev,
                    "deviation":score,
                    "reason":("체온낮음" if not pd.isna(r["rectal_temp"]) and r["rectal_temp"]<tmin else ("체온높음" if not pd.isna(r["rectal_temp"]) and r["rectal_temp"]>tmax else "결측")),
                })
        return pd.DataFrame(rows)

    def _gen_dummy(self):
        now=datetime.now().replace(second=0,microsecond=0)
        start=now-timedelta(minutes=self.minutes)
        idx=pd.date_range(start,now,freq=self.freq)
        rec=[]
        pig_map=[]
        pid=1
        for c in range(1,self.chambers+1):
            for k in range(self.ppc):
                pig_map.append((c,pid,self.rng.uniform(25,110)))
                pid+=1
        for ts in idx:
            for c,p,w in pig_map:
                tmin,tmax=self._normal_range(w)
                mu=(tmin+tmax)/2
                sigma=0.25
                temp=float(self.rng.normal(mu,sigma))
                if self.rng.uniform()<0.03: temp=np.nan
                if self.rng.uniform()<0.04:
                    if self.rng.uniform()<0.5: temp=mu-0.9*self.rng.uniform(1,1.6)
                    else: temp=mu+0.9*self.rng.uniform(1,1.6)
                rec.append({"timestamp":ts,"chamber":c,"pig_id":p,"weight_kg":round(w,2),"rectal_temp":(None if pd.isna(temp) else round(temp,2))})
        return pd.DataFrame(rec)

    def run_test(self):
        sig=self._gen_dummy()
        alerts=self._eval_df(sig)
        sig.to_csv(self.out_signals,index=False,encoding="utf-8-sig")
        alerts.to_csv(self.out_alerts,index=False,encoding="utf-8-sig")
        return sig,alerts

    def run_live(self,input_csv):
        df=pd.read_csv(input_csv)
        df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
        df=df.sort_values("timestamp")
        need=["timestamp","chamber","pig_id","weight_kg","rectal_temp"]
        for k in need:
            if k not in df.columns: raise SystemExit(f"missing column: {k}")
        alerts=self._eval_df(df)
        alerts.to_csv(self.out_alerts,index=False,encoding="utf-8-sig")
        return df,alerts

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["test","live"],default="test")
    ap.add_argument("--input_csv",default=None)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--freq",type=int,default=60)
    ap.add_argument("--minutes",type=int,default=720)
    ap.add_argument("--chambers",type=int,default=4)
    ap.add_argument("--pigs_per_chamber",type=int,default=6)
    ap.add_argument("--out_signals",default="signals.csv")
    ap.add_argument("--out_alerts",default="alerts.csv")
    args=ap.parse_args()
    eng=PigHealthRuleEngine(mode=args.mode,seed=args.seed,freq_min=args.freq,minutes=args.minutes,chambers=args.chambers,pigs_per_chamber=args.pigs_per_chamber,out_signals=args.out_signals,out_alerts=args.out_alerts)
    if args.mode=="test":
        sig,alerts=eng.run_test()
    else:
        if not args.input_csv: raise SystemExit("need --input_csv for live mode")
        sig,alerts=eng.run_live(args.input_csv)
    print(json.dumps({
        "mode":args.mode,
        "signals_rows":(0 if sig is None else len(sig)),
        "alerts_rows":(0 if alerts is None else len(alerts)),
        "out_signals":args.out_signals,
        "out_alerts":args.out_alerts
    },ensure_ascii=False))
