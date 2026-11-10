"""Baseline: predict Riot's primary champion class from per-champion average stats.
usage: python3 -I baseline.py <match_csv> <champion_json> <outdir> [nofinal]
(add "nofinal" as the 4th argument to drop the 13 final_* stats)
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
warnings.filterwarnings("ignore")

csv_path, json_path, outdir = sys.argv[1:4]
MIN_GAMES, N_SEEDS, N_SHUF, N_BOOT = 100, 10, 20, 5000

# ---------- data ----------
df = pd.read_csv(csv_path)
champs = json.load(open(json_path, encoding="utf-8"))["data"]
key2 = {int(v["key"]): v for v in champs.values()}
print(f"rows={len(df)} cols={df.shape[1]} games={df.game_id.nunique()} champions={df.champion_id.nunique()}")

mins = df.duration / 60.0
f = pd.DataFrame({"champion_id": df.champion_id})
for c in ["kills", "deaths", "assists", "damage_taken", "vision_score"]:
    f[c + "_pm"] = df[c] / mins          # per-minute, before averaging
for c in ["kda_ratio", "kill_participation", "gold_per_min", "damage_per_min", "damage_champ_per_min"]:
    f[c] = df[c]
fin = [c for c in df.columns if c.startswith("final_")] if not (len(sys.argv) > 4 and sys.argv[4] == "nofinal") else []
for c in fin:
    f[c] = df[c]
inputs = [c for c in f.columns if c != "champion_id"]
print("inputs:", len(inputs))
print("non-finite cells:", int((~np.isfinite(f[inputs].to_numpy(float))).sum()), "| NaN:", int(f[inputs].isna().sum().sum()))

g = f.groupby("champion_id")
X_all = g[inputs].mean()
n_games = g.size()
keep = n_games[n_games >= MIN_GAMES].index
dropped = n_games[n_games < MIN_GAMES]
print(f"champions kept (>= {MIN_GAMES} games): {len(keep)}; dropped: {len(dropped)} ->",
      [(key2[i]['name'], int(n)) for i, n in dropped.items()])
X_all = X_all.loc[keep]
y_all = pd.Series({i: key2[i]["tags"][0] for i in keep})
names = pd.Series({i: key2[i]["name"] for i in keep})
X = X_all.to_numpy(float); y = y_all.to_numpy()
classes, counts = np.unique(y, return_counts=True)
print("N =", len(y), "| class counts:", dict(zip(classes, counts.tolist())), "| largest-class share:", round(counts.max()/len(y), 3))

# ---------- models ----------
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21]
C_GRID = list(np.logspace(-3, 2, 11))
def make(kind, seed):
    inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1000)
    if kind == "knn":
        pipe = Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())])
        return GridSearchCV(pipe, {"m__n_neighbors": K_GRID}, cv=inner, scoring="accuracy")
    if kind == "logit":
        pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(penalty="l2", max_iter=5000))])
        return GridSearchCV(pipe, {"m__C": C_GRID}, cv=inner, scoring="accuracy")
    return DummyClassifier(strategy="most_frequent")

def run_cv(Xm, ym, seed, kinds=("floor", "knn", "logit")):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = {k: np.zeros(len(ym), bool) for k in kinds}
    chosen = {"knn": [], "logit": []}
    for tr, te in outer.split(Xm, ym):
        for k in kinds:
            m = make(k, seed).fit(Xm[tr], ym[tr])
            oof[k][te] = (m.predict(Xm[te]) == ym[te])
            if k == "knn": chosen["knn"].append(m.best_params_["m__n_neighbors"])
            if k == "logit": chosen["logit"].append(m.best_params_["m__C"])
    return oof, chosen

# ---------- real labels: 10 seeds x stratified 5-fold ----------
oof_by = {k: [] for k in ("floor", "knn", "logit")}
chosen_all = {"knn": [], "logit": []}
for s in range(N_SEEDS):
    oof, ch = run_cv(X, y, s)
    for k in oof_by: oof_by[k].append(oof[k])
    for k in ch: chosen_all[k] += ch[k]
    print(f"seed {s}: " + "  ".join(f"{k}={oof[k].mean():.3f}" for k in oof_by))

res = {"N": int(len(y)), "n_inputs": len(inputs), "classes": dict(zip(classes.tolist(), counts.tolist()))}
rng = np.random.default_rng(12345)
for k in oof_by:
    M = np.array(oof_by[k], float)              # seeds x champions (1 = correct)
    per_seed = M.mean(1)
    per_champ = M.mean(0)                        # fraction of seeds this champion was classified correctly
    boot = np.array([per_champ[rng.integers(0, len(y), len(y))].mean() for _ in range(N_BOOT)])
    res[k] = dict(acc=float(per_seed.mean()), seed_sd=float(per_seed.std(ddof=1)),
                  seed_min=float(per_seed.min()), seed_max=float(per_seed.max()),
                  boot_lo=float(np.percentile(boot, 2.5)), boot_hi=float(np.percentile(boot, 97.5)),
                  boot_se=float(boot.std(ddof=1)))
    print(f"{k:6s} acc={res[k]['acc']:.3f}  95% boot CI [{res[k]['boot_lo']:.3f}, {res[k]['boot_hi']:.3f}]  seed SD={res[k]['seed_sd']:.3f}  seeds [{res[k]['seed_min']:.3f}, {res[k]['seed_max']:.3f}]")
res["chosen_k_counts"] = {int(k): int(v) for k, v in zip(*np.unique(chosen_all["knn"], return_counts=True))}
res["chosen_C_counts"] = {f"{k:.3g}": int(v) for k, v in zip(*np.unique(chosen_all["logit"], return_counts=True))}
res["chosen_k_median"] = float(np.median(chosen_all["knn"]))
print("kNN chosen k counts (50 outer fits):", res["chosen_k_counts"])
print("logit chosen C counts:", res["chosen_C_counts"])

# per-class recall for kNN (pooled over seeds)
pred_ok = np.array(oof_by["knn"], float).mean(0)
res["knn_recall_by_class"] = {c: float(pred_ok[y == c].mean()) for c in classes}
print("kNN per-class accuracy:", {c: round(v, 3) for c, v in res["knn_recall_by_class"].items()})

# ---------- shuffled-label reference (same nested pipeline) ----------
sh = {"knn": [], "logit": []}
for i in range(N_SHUF):
    ys = np.random.default_rng(777 + i).permutation(y)    # keeps class sizes
    oof, _ = run_cv(X, ys, seed=500 + i, kinds=("knn", "logit"))
    for k in sh: sh[k].append(oof[k].mean())
for k in sh:
    a = np.array(sh[k])
    res[f"shuffled_{k}"] = dict(mean=float(a.mean()), p95=float(np.percentile(a, 95)), max=float(a.max()))
    print(f"shuffled-label {k}: mean={a.mean():.3f}  95th pct={np.percentile(a,95):.3f}  max={a.max():.3f}  (n={N_SHUF})")

# save
import os; os.makedirs(outdir, exist_ok=True)
json.dump(res, open(os.path.join(outdir, "results.json"), "w", encoding="utf-8"), indent=2)
pd.DataFrame({"champion": names.values, "class": y, "games": n_games.loc[keep].values,
              "knn_correct_frac": np.array(oof_by["knn"], float).mean(0),
              "logit_correct_frac": np.array(oof_by["logit"], float).mean(0)}).to_csv(os.path.join(outdir, "per_champion.csv"), index=False)
print("saved to", outdir)