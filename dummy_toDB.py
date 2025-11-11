# facility_db_5min.py
import os, time, math, hashlib, argparse
import numpy as np
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import DateTime, Float
from sqlalchemy.dialects.mysql import insert as mysql_insert

def _rng(seed):
    h = hashlib.sha256(seed.encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))

def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))

def _phase(ts):
    t = ts.hour * 3600 + ts.minute * 60 + ts.second
    return (t % 86400) / 86400.0

def _row_for(ts_utc):
    r = _rng(ts_utc.strftime("%Y-%m-%d %H:%M"))
    p = _phase(ts_utc)
    outside_temp = 15 + 7 * math.sin(2 * math.pi * p - 0.5) + r.normal(0, 0.6)
    outside_humidity = 60 + 10 * math.sin(2 * math.pi * (p + 0.25)) + r.normal(0, 1.5)
    inside_temp = 25 + 0.6 * math.sin(2 * math.pi * (p + 0.1)) + 0.05 * (outside_temp - 15) + r.normal(0, 0.15)
    inside_humidity = 46 + 4 * math.sin(2 * math.pi * (p + 0.35)) + 0.1 * (outside_humidity - 60) + r.normal(0, 0.8)
    co2_base = 820 + 150 * max(0, math.sin(2 * math.pi * (p - 0.05))) + r.normal(0, 25)
    nh3_base = 6 + 1.2 * max(0, math.sin(2 * math.pi * (p + 0.3))) + r.normal(0, 0.25)
    evaporation = 65 + 12 * math.sin(2 * math.pi * (p + 0.2)) - 0.15 * outside_humidity + r.normal(0, 2.2)
    ventilation = 2.3 + 0.4 * max(0, (inside_temp - 25)) + 0.3 * max(0, (inside_humidity - 48) / 10) + r.normal(0, 0.05)
    ventilation = _clip(ventilation, 1.5, 3.2)
    fan_rpm = 700 + 450 * (ventilation - 2.0) + r.normal(0, 60)
    fan_rpm = _clip(fan_rpm, 200, 1500)
    fan_power_w = 90 + 0.05 * fan_rpm + r.normal(0, 6)
    heating_value = 850 - 12 * (outside_temp - 10) + r.normal(0, 20)
    heating_value = _clip(heating_value, 700, 1050)
    T_0_5 = 25.5 + 0.7 * math.sin(2 * math.pi * (p + 0.08)) + r.normal(0, 0.12)
    T_1_5 = 25.4 + 0.6 * math.sin(2 * math.pi * (p + 0.1)) + r.normal(0, 0.10)
    RH_0_5 = 35 + 6 * math.sin(2 * math.pi * (p + 0.22)) + r.normal(0, 1.2)
    RH_1_5 = 19 + 5 * math.sin(2 * math.pi * (p + 0.28)) + r.normal(0, 1.1)
    CO2_0_5 = co2_base + r.normal(0, 15)
    CO2_1_5 = co2_base + 40 + r.normal(0, 15)
    NH3_0_5 = 11 + 0.6 * math.sin(2 * math.pi * (p + 0.33)) + r.normal(0, 0.2)
    NH3_1_5 = nh3_base + r.normal(0, 0.15)
    return {
        "ts": ts_utc.replace(second=0, microsecond=0),
        "evaporation": float(evaporation),
        "T_0_5": float(T_0_5),
        "RH_0_5": float(RH_0_5),
        "CO2_0_5": float(CO2_0_5),
        "NH3_0_5": float(NH3_0_5),
        "T_1_5": float(T_1_5),
        "RH_1_5": float(RH_1_5),
        "CO2_1_5": float(CO2_1_5),
        "NH3_1_5": float(NH3_1_5),
        "fan_rpm": float(fan_rpm),
        "fan_power_w": float(fan_power_w),
        "ventilation": float(ventilation),
        "heating_value": float(heating_value),
        "outside_temp": float(outside_temp),
        "inside_temp": float(inside_temp),
        "outside_humidity": float(outside_humidity),
        "inside_humidity": float(inside_humidity),
    }

def _engine():
    u = os.getenv("DB_USER", "")
    p = os.getenv("DB_PASS", "")
    h = os.getenv("DB_HOST", "127.0.0.1")
    o = os.getenv("DB_PORT", "3306")
    d = os.getenv("DB_NAME", "test")
    url = f"mysql+mysqlconnector://{u}:{p}@{h}:{o}/{d}"
    return create_engine(url, pool_pre_ping=True, future=True)

def _table(meta, name):
    return Table(
        name, meta,
        Column("ts", DateTime, primary_key=True),
        Column("evaporation", Float),
        Column("T_0_5", Float),
        Column("RH_0_5", Float),
        Column("CO2_0_5", Float),
        Column("NH3_0_5", Float),
        Column("T_1_5", Float),
        Column("RH_1_5", Float),
        Column("CO2_1_5", Float),
        Column("NH3_1_5", Float),
        Column("fan_rpm", Float),
        Column("fan_power_w", Float),
        Column("ventilation", Float),
        Column("heating_value", Float),
        Column("outside_temp", Float),
        Column("inside_temp", Float),
        Column("outside_humidity", Float),
        Column("inside_humidity", Float),
        extend_existing=True
    )

def _upsert(conn, tbl, rec):
    stmt = mysql_insert(tbl).values(rec)
    upd = {k: stmt.inserted[k] for k in rec.keys() if k != "ts"}
    conn.execute(stmt.on_duplicate_key_update(**upd))

def _floor5(ts):
    m = ts.minute - (ts.minute % 5)
    return ts.replace(minute=m, second=0, microsecond=0)

def _sleep_to_next_5m(now):
    base = _floor5(now)
    nxt = base + timedelta(minutes=5)
    return max(1.0, (nxt - now).total_seconds())

def run(once=False):
    eng = _engine()
    meta = MetaData()
    tbl = _table(meta, os.getenv("TABLE_NAME", "facilities_metrics_5m"))
    meta.create_all(eng)
    with eng.begin() as conn:
        if once:
            now = datetime.now(timezone.utc)
            ts = _floor5(now)
            rec = _row_for(ts)
            _upsert(conn, tbl, rec)
            return
        while True:
            now = datetime.now(timezone.utc)
            ts = _floor5(now)
            rec = _row_for(ts)
            _upsert(conn, tbl, rec)
            time.sleep(_sleep_to_next_5m(datetime.now(timezone.utc)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    run(once=args.once)
