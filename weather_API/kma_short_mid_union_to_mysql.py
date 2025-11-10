import os, requests, datetime as dt, sqlalchemy as sa, urllib.parse, time
import pandas as pd
import numpy as np

try:
    requests.packages.urllib3.util.connection.HAS_IPV6 = False
except Exception:
    pass

raw_key=(os.environ.get("KMA_KEY") or "")
KEY=raw_key.replace("\r","").replace("\n","").strip()
DB=os.environ.get("DB_URL")
REG=os.environ.get("KMA_REG","11B00000")
REG_TEMP=os.environ.get("KMA_TEMP_REG") or REG
LAND_URL="https://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"
TEMP_URL="https://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa"
NX=int(os.environ.get("KMA_NX","60"))
NY=int(os.environ.get("KMA_NY","127"))
VILAGE_URL="https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
HDR={"Accept":"application/json,*/*","User-Agent":"Mozilla/5.0"}

def build_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s=requests.Session()
    r=Retry(total=6,connect=4,read=4,backoff_factor=1.5,status_forcelist=[429,500,502,503,504],allowed_methods=frozenset(["GET"]))
    s.mount("https://",HTTPAdapter(max_retries=r,pool_connections=50,pool_maxsize=50))
    s.headers.update(HDR)
    return s

def call(s,url,params,timeout=(10,60)):
    key_plain=params["serviceKey"]
    for i,k in enumerate([key_plain,urllib.parse.quote(key_plain,safe="")],start=1):
        params["serviceKey"]=k
        try:
            r=s.get(url,params=params,timeout=timeout)
            if r.status_code==200 and r.text.strip().lower()!="forbidden":
                return r.json()
        except Exception:
            time.sleep(0.6*i)
    raise RuntimeError(f"Request failed for {url}")

def pick_base_short_candidates():
    kst=dt.datetime.utcnow()+dt.timedelta(hours=9)
    if os.environ.get("KMA_BASE_DATE") and os.environ.get("KMA_BASE_TIME"):
        bd=os.environ["KMA_BASE_DATE"].strip(); bt=os.environ["KMA_BASE_TIME"].strip()
        return [(bd,bt)]
    valid=[2,5,8,11,14,17,20,23]
    t=kst-dt.timedelta(minutes=70)
    t=t.replace(minute=0,second=0,microsecond=0)
    c=[]
    for _ in range(18):
        if t.hour in valid:
            c.append((t.strftime("%Y%m%d"),t.strftime("%H%M")))
        t=t-dt.timedelta(hours=1)
    return c

def fetch_short(s):
    cands=pick_base_short_candidates()
    last=None
    for bd,bt in cands:
        p={"serviceKey":KEY,"dataType":"JSON","numOfRows":1000,"pageNo":1,"base_date":bd,"base_time":bt,"nx":NX,"ny":NY}
        try:
            j=call(s,VILAGE_URL,p)
            it=(((j.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
            if it:
                return it,f"{bd}{bt}"
        except Exception as e:
            last=e
        time.sleep(0.5)
    if last: raise last
    raise RuntimeError("short-term fetch failed")

def map_weather(sky,pty):
    if pty in [1,2,5,6]: return "비"
    if pty in [3,7]: return "눈"
    if pty==4: return "소나기"
    if sky==1: return "맑음"
    if sky==3: return "구름많음"
    if sky==4: return "흐림"
    return "맑음"

def to_rows_short(items,base_tm,reg_id):
    base_dt=dt.datetime.strptime(base_tm,"%Y%m%d%H%M")
    df=pd.DataFrame(items)
    df["fcstValue"]=pd.to_numeric(df["fcstValue"],errors="coerce")
    df["fcst_dt"]=pd.to_datetime(df["fcstDate"]+df["fcstTime"],format="%Y%m%d%H%M",errors="coerce")
    df["day"]=df["fcst_dt"].dt.date
    pv=df.pivot_table(index="fcst_dt",columns="category",values="fcstValue",aggfunc="first")
    pv=pv.resample("h").ffill()
    days=sorted(df["day"].unique())[:3]
    rows=[]
    for day in days:
        dstr=pd.to_datetime(str(day)).date().isoformat()
        day_data=pv[pv.index.date==day]
        if day_data.empty: continue
        tmin=day_data.get("TMN",pd.Series(dtype=float)).min()
        tmax=day_data.get("TMX",pd.Series(dtype=float)).max()
        if pd.isna(tmin):
            tmps=day_data.get("TMP",pd.Series(dtype=float))
            if not tmps.empty: tmin=tmps.min()
        if pd.isna(tmax):
            tmps=day_data.get("TMP",pd.Series(dtype=float))
            if not tmps.empty: tmax=tmps.max()
        am=day_data.between_time("06:01","12:00")
        pm=day_data.between_time("12:01","18:00")
        pop_am=am.get("POP",pd.Series(dtype=float)).max()
        pop_pm=pm.get("POP",pd.Series(dtype=float)).max()
        wf_am_sky=(am.get("SKY",pd.Series(dtype=float)).mode().iloc[0] if not am.get("SKY",pd.Series(dtype=float)).dropna().empty else 1)
        wf_am_pty=(am.get("PTY",pd.Series(dtype=float)).mode().iloc[0] if not am.get("PTY",pd.Series(dtype=float)).dropna().empty else 0)
        wf_pm_sky=(pm.get("SKY",pd.Series(dtype=float)).mode().iloc[0] if not pm.get("SKY",pd.Series(dtype=float)).dropna().empty else 1)
        wf_pm_pty=(pm.get("PTY",pd.Series(dtype=float)).mode().iloc[0] if not pm.get("PTY",pd.Series(dtype=float)).dropna().empty else 0)
        def to_int(x):
            return None if pd.isna(x) else int(round(float(x)))
        rows.append({
            "reg_id":reg_id,
            "fcst_date":dstr,
            "wf_am":map_weather(int(wf_am_sky),int(wf_am_pty)),
            "wf_pm":map_weather(int(wf_pm_sky),int(wf_pm_pty)),
            "pop_am":to_int(pop_am),
            "pop_pm":to_int(pop_pm),
            "tmin":to_int(tmin),
            "tmax":to_int(tmax),
            "tmfc":base_dt.strftime("%Y-%m-%d %H:%M:%S")
        })
    return rows

def _nz(x):
    return None if x in (None,"","-") else x

def pick_tmfc_list():
    kst=dt.datetime.utcnow()+dt.timedelta(hours=9)
    def tmfc_at(d,h): return f"{d:%Y%m%d}{h:02d}00"
    if kst.hour<6: cur=tmfc_at((kst-dt.timedelta(days=1)).date(),18)
    elif kst.hour<18: cur=tmfc_at(kst.date(),6)
    else: cur=tmfc_at(kst.date(),18)
    k=dt.datetime.strptime(cur,"%Y%m%d%H%M")
    c=[cur]
    for h in range(12,121,12):
        c.append((k-dt.timedelta(hours=h)).strftime("%Y%m%d%H%M"))
    return c

def fetch_mid_one(s,reg_land,reg_temp,tmFc):
    p_land={"serviceKey":KEY,"dataType":"JSON","pageNo":1,"numOfRows":10,"regId":reg_land,"tmFc":tmFc}
    p_temp={"serviceKey":KEY,"dataType":"JSON","pageNo":1,"numOfRows":10,"regId":reg_temp,"tmFc":tmFc}
    j1=call(s,LAND_URL,p_land)
    j2=call(s,TEMP_URL,p_temp)
    it1=(((j1.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
    it2=(((j2.get("response") or {}).get("body") or {}).get("items") or {}).get("item") or []
    if not it1 or not it2:
        return []
    land_item=it1[0]; temp_item=it2[0]
    return to_rows_mid(land_item,temp_item,tmFc)

def fetch_mid_multi(s,reg_land,reg_temp):
    rows=[]
    for tm in pick_tmfc_list():
        try:
            r=fetch_mid_one(s,reg_land,reg_temp,tm)
            if r: rows.append(r)
        except Exception:
            continue
    return rows

def to_rows_mid(land_item,temp_item,tmFc):
    base_dt=dt.datetime.strptime(tmFc,"%Y%m%d%H%M")
    rows=[]
    def to_int(x):
        x=_nz(x)
        if x is None: return None
        try: return int(x)
        except: return None
    for d in range(3,10):
        day=(base_dt.date()+dt.timedelta(days=d))
        dstr=day.isoformat()
        if d<=7:
            wf_am=_nz(land_item.get(f"wf{d}Am"))
            wf_pm=_nz(land_item.get(f"wf{d}Pm"))
            pa=_nz(land_item.get(f"rnSt{d}Am"))
            pp=_nz(land_item.get(f"rnSt{d}Pm"))
        else:
            wf=_nz(land_item.get(f"wf{d}"))
            rn=_nz(land_item.get(f"rnSt{d}"))
            wf_am=wf; wf_pm=wf; pa=rn; pp=rn
        tmin=_nz(temp_item.get(f"taMin{d}"))
        tmax=_nz(temp_item.get(f"taMax{d}"))
        rows.append({
            "reg_id":land_item.get("regId"),
            "fcst_date":dstr,
            "wf_am":wf_am,
            "wf_pm":wf_pm,
            "pop_am":to_int(pa),
            "pop_pm":to_int(pp),
            "tmin":to_int(tmin),
            "tmax":to_int(tmax),
            "tmfc":base_dt.strftime("%Y-%m-%d %H:%M:%S")
        })
    return rows

def merge_by_date(rows_sets):
    d={}
    fields=["wf_am","wf_pm","pop_am","pop_pm","tmin","tmax"]
    for rows in rows_sets:
        for r in rows:
            k=(r["reg_id"],r["fcst_date"])
            score=sum(1 for f in fields if r.get(f) is not None)
            if k not in d:
                d[k]=(r,score)
            else:
                cur,cur_score=d[k]
                if score>cur_score or (score==cur_score and r["tmfc"]>cur["tmfc"]):
                    base=r.copy()
                    for f in fields:
                        if base.get(f) is None and cur.get(f) is not None:
                            base[f]=cur[f]
                    d[k]=(base, max(score,cur_score))
                else:
                    cur2=cur.copy()
                    for f in fields:
                        if cur2.get(f) is None and r.get(f) is not None:
                            cur2[f]=r[f]
                    d[k]=(cur2, max(score,cur_score))
    return [d[k][0] for k in sorted(d.keys(),key=lambda x:x[1])]

def overlay_short(mid_rows,short_rows):
    idx={(r["reg_id"],r["fcst_date"]):r for r in mid_rows}
    for r in short_rows:
        k=(r["reg_id"],r["fcst_date"])
        if k in idx:
            cur=idx[k]
            for f in ["wf_am","wf_pm","pop_am","pop_pm","tmin","tmax"]:
                if r.get(f) is not None:
                    cur[f]=r[f]
            if r.get("tmfc","")>cur.get("tmfc",""):
                cur["tmfc"]=r["tmfc"]
        else:
            idx[k]=r
    return list(idx.values())

def final_fill(rows):
    df=pd.DataFrame(rows)
    if df.empty:
        return []
    df=df.sort_values(["reg_id","fcst_date"])
    for c in ["pop_am","pop_pm"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype(int)
    df["wf_am"]=df["wf_am"].fillna(df["wf_pm"])
    df["wf_pm"]=df["wf_pm"].fillna(df["wf_am"])
    df["wf_am"]=df["wf_am"].fillna("맑음")
    df["wf_pm"]=df["wf_pm"].fillna("맑음")
    for c in ["tmin","tmax"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
        df[c]=df.groupby("reg_id")[c].ffill().bfill().fillna(0).astype(int)
    df["tmfc"]=df["tmfc"].fillna(dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    return df.to_dict("records")

def upsert_combined(rows,e):
    if not rows: return 0
    clean=[]
    for r in rows:
        clean.append({k:(None if pd.isna(v) else v) for k,v in r.items()})
    sql=("INSERT INTO mid_land_fcst(reg_id,fcst_date,wf_am,wf_pm,pop_am,pop_pm,tmin,tmax,tmfc) "
         "VALUES(:reg_id,:fcst_date,:wf_am,:wf_pm,:pop_am,:pop_pm,:tmin,:tmax,:tmfc) "
         "ON DUPLICATE KEY UPDATE "
         "wf_am=VALUES(wf_am),wf_pm=VALUES(wf_pm),pop_am=VALUES(pop_am),pop_pm=VALUES(pop_pm),"
         "tmin=VALUES(tmin),tmax=VALUES(tmax),tmfc=VALUES(tmfc)")
    with e.begin() as cx:
        cx.execute(sa.text(sql),clean)
    return len(clean)

def refresh_delete_mid(e, reg_id, dates, refresh_all=False):
    if refresh_all:
        with e.begin() as cx:
            cx.execute(sa.text("TRUNCATE TABLE mid_land_fcst"))
        return
    if not dates:
        return
    s=min(dates); e_d=max(dates)
    with e.begin() as cx:
        cx.execute(sa.text("DELETE FROM mid_land_fcst WHERE reg_id=:r AND fcst_date>=:s AND fcst_date<=:e"),
                   {"r":reg_id,"s":s,"e":e_d})

def main():
    if not KEY or not DB: raise SystemExit("set KMA_KEY and DB_URL")
    s=build_session()
    short_items,short_base=fetch_short(s)
    rows_short=to_rows_short(short_items,short_base,REG)
    mid_sets=fetch_mid_multi(s,REG,REG_TEMP)
    rows_mid=merge_by_date(mid_sets)
    merged=overlay_short(rows_mid,rows_short)
    merged=final_fill(merged)
    e=sa.create_engine(DB)
    refresh_all=os.environ.get("MID_REFRESH_ALL","0")=="1"
    dates=sorted({r["fcst_date"] for r in merged})
    refresh_delete_mid(e, REG, dates, refresh_all)
    n=upsert_combined(merged,e)
    print({"deleted":"ALL" if refresh_all else f"{dates[0]}~{dates[-1]}" if dates else "NONE",
           "upserted":n,"total_days":len(merged),"reg_id":REG,"nx":NX,"ny":NY})

if __name__=="__main__":
    main()
