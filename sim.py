import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
import math
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
BETA0, BETA1, BETA2 = -0.5, 1.2, 0.8   # quadratic DGP on Normal X
N_VALUES  = [15, 50, 200]
M_FACTORS = [0.25, 0.5, 0.75]
D_FRACTIONS = [0.25, 0.5, 0.75]
B         = 400
SIM_REPS  = 100
N_JOBS    = -1  # -1 = all cores

# ── Helpers ───────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_data(n, seed=None):
    """
    X ~ Normal(0, 1)
    True DGP: logit(p) = beta0 + beta1*X + beta2*X^2  (quadratic)
    
    Why discretisation helps here:
      - Linear logistic regression on X can only fit a monotone sigmoid —
        it cannot capture the U-shaped / accelerating quadratic relationship.
      - Binning X into quantile intervals lets each bin get its own coefficient,
        approximating the nonlinear step function without knowing the true form.
    """
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal(size=n)
    p   = sigmoid(BETA0 + BETA1 * X + BETA2 * X ** 2)
    Y   = rng.binomial(1, p)
    return X.reshape(-1, 1), Y

def true_auc(n_ref=20_000):
    X, Y = generate_data(n_ref, seed=0)
    p = sigmoid(BETA0 + BETA1 * X.ravel() + BETA2 * X.ravel() ** 2)
    return roc_auc_score(Y, p)

# ── Models ────────────────────────────────────────────────────────────────────
def fit_model3(X_tr, Y_tr, X_te):
    """
    M3: Simple logistic regression on raw X.
    Can only fit a linear logit → misses the quadratic signal.
    """
    counts = np.bincount(Y_tr)
    if len(counts) < 2 or counts.min() == 0:
        return None
    w   = {0: 1.0, 1: counts[0] / counts[1]}
    clf = LogisticRegression(class_weight=w, solver="lbfgs", max_iter=300)
    clf.fit(X_tr, Y_tr)
    return clf.predict_proba(X_te)[:, 1]

def fit_model4(X_tr, Y_tr, X_te, max_bins=4):
    """
    M4: Quantile-bin X then fit logistic regression on one-hot bins.
    Binning X ~ N(0,1) into quantile intervals captures the nonlinear
    (quadratic) DGP without knowing the true functional form.
    Each bin gets its own intercept-like coefficient, approximating
    the true quadratic step-wise.
    """
    counts = np.bincount(Y_tr)
    if len(counts) < 2 or counts.min() == 0:
        return None

    n_bins = min(max_bins, len(np.unique(X_tr)))
    X_tr_d = X_te_d = None
    while n_bins >= 2:
        kbd = KBinsDiscretizer(
            n_bins=n_bins, encode="onehot-dense",
            strategy="quantile", quantile_method="averaged_inverted_cdf",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                X_tr_d = kbd.fit_transform(X_tr)
                X_te_d = kbd.transform(X_te)
                break
            except ValueError:
                n_bins -= 1

    if X_tr_d is None or X_tr_d.shape[1] < 2:
        return None

    clf = LogisticRegression(solver="lbfgs", max_iter=300)
    clf.fit(X_tr_d, Y_tr)
    return clf.predict_proba(X_te_d)[:, 1]

def compute_auc(probs, Y):
    if probs is None or len(np.unique(Y)) < 2:
        return np.nan
    return roc_auc_score(Y, probs)

# ── Resampling ────────────────────────────────────────────────────────────────
def bootstrap_auc(X, Y, m, B, model_fn, rng):
    aucs, n = [], len(Y)
    for _ in range(B):
        idx = rng.choice(n, size=m, replace=True)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) == 0 or len(np.unique(Y[idx])) < 2:
            continue
        probs = model_fn(X[idx], Y[idx], X[oob])
        auc   = compute_auc(probs, Y[oob])
        if not np.isnan(auc):
            aucs.append(auc)
    return np.array(aucs)

def jackknife_auc(X, Y, d, model_fn, max_reps=300, rng=None):
    """Delete-d jackknife. d = floor(n * fraction)."""
    n, all_idx, aucs = len(Y), np.arange(len(Y)), []
    if d == 1:
        subsets = [np.delete(all_idx, i) for i in range(n)]
    else:
        total = int(math.comb(n, d)) if n <= 30 else max_reps + 1
        if total <= max_reps:
            subsets = [np.array(list(c)) for c in combinations(range(n), n - d)]
        else:
            subsets = [
                np.sort(rng.choice(n, size=n - d, replace=False))
                for _ in range(max_reps)
            ]
    for keep in subsets:
        drop = np.setdiff1d(all_idx, keep)
        X_tr, Y_tr = X[keep], Y[keep]
        if len(np.unique(Y_tr)) < 2:
            continue
        probs = model_fn(X_tr, Y_tr, X[drop])
        auc   = compute_auc(probs, Y[drop])
        if not np.isnan(auc):
            aucs.append(auc)
    return np.array(aucs)

# ── Worker ────────────────────────────────────────────────────────────────────
MODELS = {
    "M3_linear": fit_model3,
    "M4_binned": fit_model4,
}

def run_rep(n, rep):
    seed = n * 10_000 + rep
    rng  = np.random.default_rng(seed)
    X, Y = generate_data(n, seed=seed)
    rows = []

    # Bootstrap
    for mf in M_FACTORS:
        m = max(int(mf * n), 5)
        for model_name, model_fn in MODELS.items():
            aucs = bootstrap_auc(X, Y, m, B, model_fn, rng)
            if len(aucs) > 0:
                rows.append({
                    "n": n, "rep": rep, "method": "bootstrap",
                    "factor": f"m=n×{mf}", "model": model_name,
                    "mean_auc": float(np.mean(aucs)),
                    "se_auc":   float(np.std(aucs)),
                })

    # Delete-d jackknife with d = floor(n * fraction)
    for frac in D_FRACTIONS:
        d = max(1, int(np.floor(frac * n)))
        if d >= n:
            continue
        label = f"d=n×{frac}"
        for model_name, model_fn in MODELS.items():
            aucs = jackknife_auc(X, Y, d, model_fn, rng=rng)
            if len(aucs) > 0:
                rows.append({
                    "n": n, "rep": rep, "method": "jackknife",
                    "factor": label, "model": model_name,
                    "mean_auc": float(np.mean(aucs)),
                    "se_auc":   float(np.std(aucs)),
                })
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_cores = multiprocessing.cpu_count()
    tasks   = [(n, rep) for n in N_VALUES for rep in range(SIM_REPS)]
    n_tasks = len(tasks)

    print(f"\nAUC Resampling Simulation")
    print(f"  DGP: X ~ N(0,1),  logit(p) = {BETA0} + {BETA1}*X + {BETA2}*X^2  (quadratic)")
    print(f"  M3: linear logistic on X  |  M4: logistic on quantile-binned X")
    print(f"  N={N_VALUES}  M_factors={M_FACTORS}  D_fractions={D_FRACTIONS}")
    print(f"  B={B}  reps={SIM_REPS}  tasks={n_tasks}  cores={n_cores}\n")

    print("Computing true AUC on 20 000 samples...")
    TRUE_AUC = true_auc()
    print(f"  True AUC = {TRUE_AUC:.4f}\n")

    t0 = time.perf_counter()

    all_rows = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_rep)(n, rep)
        for n, rep in tqdm(tasks, desc="Simulating", unit="task", dynamic_ncols=True)
    )

    wall = time.perf_counter() - t0
    print(f"\nDone in {wall:.1f}s  ({n_tasks / wall:.1f} tasks/s)\n")

    # ── Results ───────────────────────────────────────────────────────────────
    results = [row for batch in all_rows for row in batch]
    df      = pd.DataFrame(results)

    summary = (
        df.groupby(["n", "method", "factor", "model"])
        .agg(
            est_auc=("mean_auc", "mean"),
            bias   =("mean_auc", lambda x: np.mean(x) - TRUE_AUC),
            se     =("mean_auc", "std"),
            avg_se =("se_auc",   "mean"),
        )
        .reset_index()
    )
    summary["rmse"] = np.sqrt(summary["bias"] ** 2 + summary["se"] ** 2)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.width", 140)

    print(f"True AUC = {TRUE_AUC:.4f}\n")
    print(summary.sort_values(["n", "method", "factor", "model"]).to_string(index=False))

    pivot = summary.pivot_table(
        index=["n", "method", "factor"],
        columns="model",
        values=["bias", "se", "rmse"],
    ).round(4)
    print("\n── Pivot: bias / SE / RMSE  (M3_linear vs M4_binned) ──")
    print(pivot.to_string())

    # ── Advantage summary ─────────────────────────────────────────────────────
    print("\n── M4 advantage over M3 (RMSE reduction, averaged over factors) ──")
    adv = (
        summary.pivot_table(index=["n", "method", "factor"], columns="model", values="rmse")
        .assign(rmse_reduction=lambda d: d["M3_linear"] - d["M4_binned"])
        .groupby(["n", "method"])["rmse_reduction"]
        .mean()
        .reset_index()
    )
    print(adv.to_string(index=False))
# AUC Resampling Simulation
#   DGP: X ~ N(0,1),  logit(p) = -0.5 + 1.2*X + 0.8*X^2  (quadratic)
#   M3: linear logistic on X  |  M4: logistic on quantile-binned X
#   N=[15, 50, 200]  M_factors=[0.25, 0.5, 0.75]  D_fractions=[0.25, 0.5, 0.75]
#   B=400  reps=100  tasks=300  cores=4

# Computing true AUC on 20 000 samples...
#   True AUC = 0.7554

# Simulating: 100%|███████████████████████████████████████████████████████████████████████████████| 300/300 [30:39<00:00,  6.13s/task]

# Done in 1874.5s  (0.2 tasks/s)

# True AUC = 0.7554

#   n    method   factor     model  est_auc    bias     se  avg_se   rmse
#  15 bootstrap m=n×0.25 M3_linear   0.6289 -0.1264 0.1749  0.1774 0.2159
#  15 bootstrap m=n×0.25 M4_binned   0.5824 -0.1729 0.1103  0.1647 0.2051
#  15 bootstrap  m=n×0.5 M3_linear   0.6399 -0.1155 0.1835  0.1779 0.2168
#  15 bootstrap  m=n×0.5 M4_binned   0.5894 -0.1660 0.1132  0.1784 0.2009
#  15 bootstrap m=n×0.75 M3_linear   0.6533 -0.1020 0.1921  0.2008 0.2176
#  15 bootstrap m=n×0.75 M4_binned   0.6036 -0.1517 0.1266  0.2050 0.1976
#  15 jackknife d=n×0.25 M3_linear   0.6812 -0.0742 0.2152  0.3113 0.2276
#  15 jackknife d=n×0.25 M4_binned   0.6334 -0.1220 0.1712  0.2512 0.2102
#  15 jackknife  d=n×0.5 M3_linear   0.6569 -0.0985 0.1980  0.1899 0.2212
#  15 jackknife  d=n×0.5 M4_binned   0.6098 -0.1456 0.1363  0.1581 0.1994
#  15 jackknife d=n×0.75 M3_linear   0.6244 -0.1310 0.1738  0.1775 0.2177
#  15 jackknife d=n×0.75 M4_binned   0.5772 -0.1782 0.1076  0.1313 0.2081
#  50 bootstrap m=n×0.25 M3_linear   0.6664 -0.0890 0.0951  0.1318 0.1303
#  50 bootstrap m=n×0.25 M4_binned   0.6317 -0.1237 0.0692  0.0997 0.1417
#  50 bootstrap  m=n×0.5 M3_linear   0.6932 -0.0622 0.0974  0.1000 0.1155
#  50 bootstrap  m=n×0.5 M4_binned   0.6595 -0.0959 0.0763  0.0976 0.1225
#  50 bootstrap m=n×0.75 M3_linear   0.7009 -0.0545 0.0952  0.1006 0.1097
#  50 bootstrap m=n×0.75 M4_binned   0.6709 -0.0845 0.0787  0.1013 0.1155
#  50 jackknife d=n×0.25 M3_linear   0.7149 -0.0405 0.0875  0.1459 0.0964
#  50 jackknife d=n×0.25 M4_binned   0.6919 -0.0635 0.0845  0.1388 0.1057
#  50 jackknife  d=n×0.5 M3_linear   0.7057 -0.0497 0.0940  0.0884 0.1064
#  50 jackknife  d=n×0.5 M4_binned   0.6791 -0.0763 0.0787  0.0890 0.1096
#  50 jackknife d=n×0.75 M3_linear   0.6791 -0.0763 0.0974  0.1140 0.1238
#  50 jackknife d=n×0.75 M4_binned   0.6477 -0.1077 0.0725  0.0971 0.1298
# 200 bootstrap m=n×0.25 M3_linear   0.7043 -0.0511 0.0398  0.0449 0.0647
# 200 bootstrap m=n×0.25 M4_binned   0.6852 -0.0702 0.0330  0.0441 0.0776
# 200 bootstrap  m=n×0.5 M3_linear   0.7095 -0.0458 0.0357  0.0329 0.0581
# 200 bootstrap  m=n×0.5 M4_binned   0.6948 -0.0606 0.0331  0.0391 0.0691
# 200 bootstrap m=n×0.75 M3_linear   0.7097 -0.0457 0.0349  0.0405 0.0574
# 200 bootstrap m=n×0.75 M4_binned   0.6980 -0.0574 0.0343  0.0438 0.0669
# 200 jackknife d=n×0.25 M3_linear   0.7106 -0.0448 0.0345  0.0657 0.0565
# 200 jackknife d=n×0.25 M4_binned   0.7028 -0.0526 0.0365  0.0627 0.0640
# 200 jackknife  d=n×0.5 M3_linear   0.7101 -0.0453 0.0348  0.0377 0.0571
# 200 jackknife  d=n×0.5 M4_binned   0.6992 -0.0562 0.0352  0.0400 0.0663
# 200 jackknife d=n×0.75 M3_linear   0.7074 -0.0480 0.0381  0.0342 0.0613
# 200 jackknife d=n×0.75 M4_binned   0.6907 -0.0646 0.0327  0.0384 0.0725

# ── Pivot: bias / SE / RMSE  (M3_linear vs M4_binned) ──
#                             bias                rmse                  se          
# model                  M3_linear M4_binned M3_linear M4_binned M3_linear M4_binned
# n   method    factor                                                              
# 15  bootstrap m=n×0.25   -0.1264   -0.1729    0.2159    0.2051    0.1749    0.1103
#               m=n×0.5    -0.1155   -0.1660    0.2168    0.2009    0.1835    0.1132
#               m=n×0.75   -0.1020   -0.1517    0.2176    0.1976    0.1921    0.1266
#     jackknife d=n×0.25   -0.0742   -0.1220    0.2276    0.2102    0.2152    0.1712
#               d=n×0.5    -0.0985   -0.1456    0.2212    0.1994    0.1980    0.1363
#               d=n×0.75   -0.1310   -0.1782    0.2177    0.2081    0.1738    0.1076
# 50  bootstrap m=n×0.25   -0.0890   -0.1237    0.1303    0.1417    0.0951    0.0692
#               m=n×0.5    -0.0622   -0.0959    0.1155    0.1225    0.0974    0.0763
#               m=n×0.75   -0.0545   -0.0845    0.1097    0.1155    0.0952    0.0787
#     jackknife d=n×0.25   -0.0405   -0.0635    0.0964    0.1057    0.0875    0.0845
#               d=n×0.5    -0.0497   -0.0763    0.1064    0.1096    0.0940    0.0787
#               d=n×0.75   -0.0763   -0.1077    0.1238    0.1298    0.0974    0.0725
# 200 bootstrap m=n×0.25   -0.0511   -0.0702    0.0647    0.0776    0.0398    0.0330
#               m=n×0.5    -0.0458   -0.0606    0.0581    0.0691    0.0357    0.0331
#               m=n×0.75   -0.0457   -0.0574    0.0574    0.0669    0.0349    0.0343
#     jackknife d=n×0.25   -0.0448   -0.0526    0.0565    0.0640    0.0345    0.0365
#               d=n×0.5    -0.0453   -0.0562    0.0571    0.0663    0.0348    0.0352
#               d=n×0.75   -0.0480   -0.0646    0.0613    0.0725    0.0381    0.0327

# ── M4 advantage over M3 (RMSE reduction, averaged over factors) ──
#   n    method  rmse_reduction
#  15 bootstrap          0.0155
#  15 jackknife          0.0162
#  50 bootstrap         -0.0081
#  50 jackknife         -0.0062
# 200 bootstrap         -0.0111
# 200 jackknife         -0.0093
#   AUC Resampling Simulation
#   N=[15, 50, 200]  M_factors=[0.25, 0.5, 0.75]  D_fractions=[0.25, 0.5, 0.75]
#   B=400  reps=100  tasks=300  cores=4

# Computing true AUC on 20 000 samples...
#   True AUC = 0.8806

# True AUC = 0.8806

#   n    method   factor model  est_auc    bias     se  avg_se   rmse
#  15 bootstrap m=n×0.25    M4   0.7185 -0.1620 0.1394  0.1724 0.2138
#  15 bootstrap  m=n×0.5    M4   0.7119 -0.1686 0.1328  0.2032 0.2147
#  15 bootstrap m=n×0.75    M4   0.7277 -0.1529 0.1393  0.2151 0.2068
#  15 jackknife d=n×0.25    M4   0.7699 -0.1107 0.1591  0.2055 0.1938
#  15 jackknife  d=n×0.5    M4   0.7506 -0.1299 0.1494  0.1498 0.1980
#  15 jackknife d=n×0.75    M4   0.7030 -0.1775 0.1398  0.1466 0.2259
#  50 bootstrap m=n×0.25    M4   0.7530 -0.1275 0.0779  0.1105 0.1495
#  50 bootstrap  m=n×0.5    M4   0.7834 -0.0972 0.0738  0.1002 0.1220
#  50 bootstrap m=n×0.75    M4   0.7974 -0.0831 0.0707  0.1021 0.1091
#  50 jackknife d=n×0.25    M4   0.8209 -0.0596 0.0731  0.1389 0.0944
#  50 jackknife  d=n×0.5    M4   0.8010 -0.0796 0.0705  0.0917 0.1063
#  50 jackknife d=n×0.75    M4   0.7584 -0.1222 0.0789  0.1139 0.1454
# 200 bootstrap m=n×0.25    M4   0.8201 -0.0605 0.0316  0.0397 0.0683
# 200 bootstrap  m=n×0.5    M4   0.8299 -0.0507 0.0312  0.0364 0.0596
# 200 bootstrap m=n×0.75    M4   0.8324 -0.0482 0.0315  0.0414 0.0576
# 200 jackknife d=n×0.25    M4   0.8383 -0.0422 0.0324  0.0601 0.0532
# 200 jackknife  d=n×0.5    M4   0.8340 -0.0465 0.0323  0.0377 0.0567
# 200 jackknife d=n×0.75    M4   0.8248 -0.0557 0.0311  0.0347 0.0638

# ── Pivot: bias / SE / RMSE by model ──
#                           bias   rmse     se       
# model                       M4     M4     M4
# n   method    factor                        
# 15  bootstrap m=n×0.25 -0.1620 0.2138 0.1394
#               m=n×0.5  -0.1686 0.2147 0.1328
#               m=n×0.75 -0.1529 0.2068 0.1393
#     jackknife d=n×0.25 -0.1107 0.1938 0.1591
#               d=n×0.5  -0.1299 0.1980 0.1494
#               d=n×0.75 -0.1775 0.2259 0.1398
# 50  bootstrap m=n×0.25 -0.1275 0.1495 0.0779
#               m=n×0.5  -0.0972 0.1220 0.0738
#               m=n×0.75 -0.0831 0.1091 0.0707
#     jackknife d=n×0.25 -0.0596 0.0944 0.0731
#               d=n×0.5  -0.0796 0.1063 0.0705
#               d=n×0.75 -0.1222 0.1454 0.0789
# 200 bootstrap m=n×0.25 -0.0605 0.0683 0.0316
#               m=n×0.5  -0.0507 0.0596 0.0312
#               m=n×0.75 -0.0482 0.0576 0.0315
#     jackknife d=n×0.25 -0.0422 0.0532 0.0324
#               d=n×0.5  -0.0465 0.0567 0.0323
#               d=n×0.75 -0.0557 0.0638 0.0311