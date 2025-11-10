import json, numpy as np, xgboost as xgb

X = np.random.rand(8000, 32).astype("float32")
y = np.random.rand(8000).astype("float32")

def train(device):
    m = xgb.XGBRegressor(
        n_estimators=80, max_depth=6,
        tree_method="hist", device=device, verbosity=1
    )
    m.fit(X, y)
    cfg = m.get_booster().save_config()
    print(f"[OK] device={device}")
    print(("cuda" if "cuda" in cfg.lower() else "cpu"), "detected in config")
    print(cfg[:800], "...\n")

try:
    print("[TRY] GPU (device='cuda:0')")
    train("cuda:0")
    print("=> GPU 사용 확인")
except Exception as e:
    print("[FAIL] GPU:", e)
    print("[TRY] CPU (device='cpu')")
    train("cpu")
    print("=> CPU 사용 중")
