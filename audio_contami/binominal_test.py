"""
to_binominal_test.csv の各列について，
1 が出た回数が理論確率 p=0.05 で偶然かどうかを二項検定する（右片側：0.05 より多いか）．
多重比較補正（Bonferroni と FDR）を適用する．
"""

import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

# データ読み込み
df = pd.read_csv("to_binominal_test.csv")
df = df.apply(pd.to_numeric, errors="coerce")

p_theory = 0.05
alpha = 0.001

results = []
for col in df.columns:
    vec = df[col].dropna()
    n = len(vec)
    k = int(vec.sum())
    if n == 0:
        results.append({
            "column": col,
            "n_trials": 0,
            "n_ones": 0,
            "observed_p": None,
            "p_value": None,
        })
        continue
    observed_p = k / n
    # greater: 1 が理論確率より多く出ていないか（偶然より多いか）の検定
    res = stats.binomtest(k, n, p=p_theory, alternative="greater")
    results.append({
        "column": col,
        "n_trials": n,
        "n_ones": k,
        "observed_p": round(observed_p, 4),
        "p_value": res.pvalue,
    })

res_df = pd.DataFrame(results)
p_values = res_df["p_value"].dropna().astype(float)
n_tests = len(p_values)

# Bonferroni 補正
bonferroni_alpha = alpha / n_tests
res_df["reject_bonferroni"] = res_df["p_value"] <= bonferroni_alpha

# FDR (Benjamini–Hochberg) 補正
res_df["p_fdr"] = None
res_df.loc[p_values.index, "p_fdr"] = false_discovery_control(p_values.values)
res_df["reject_fdr"] = res_df["p_fdr"] <= alpha

# 出力
print("理論確率 p = 0.05 の二項検定（右片側：1 が偶然より多いか），多重比較補正あり\n")
print(res_df.to_string())
print("\n--- 有意水準 ---")
print(f"有意水準 α = {alpha}")
print(f"Bonferroni 補正後 α = {alpha}/{n_tests} = {bonferroni_alpha:.6f}")
print("\n--- 解釈 ---")
print("reject_bonferroni / reject_fdr が True → 偶然とは言えず，1 の生起率は 0.05 より有意に高い．")

sig_b = res_df[res_df["reject_bonferroni"]]["column"].tolist()
sig_f = res_df[res_df["reject_fdr"]]["column"].tolist()
print(f"\nBonferroni で有意: {sig_b if sig_b else 'なし'}")
print(f"FDR で有意: {sig_f if sig_f else 'なし'}")
