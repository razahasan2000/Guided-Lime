"""
06_Statistical_Validation.py
================================
Tier 1 additions:
  - Bootstrap 95% confidence intervals over per-image fidelity AUC scores
  - Wilcoxon signed-rank tests: LIME vs Random (deletion & insertion)
  - Effect size (rank-biserial correlation)

Reads:  fidelity_metrics_results.json  (produced by run_fidelity.py)
Writes: statistical_validation_results.json
        statistical_validation_report.txt
        bootstrap_confidence_intervals.png
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ------ CONFIG ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
RESULTS_FILE   = 'fidelity_metrics_results.json'
N_BOOTSTRAP    = 1000
ALPHA          = 0.05       # significance level
SEED           = 42
OUTPUT_JSON    = 'statistical_validation_results.json'
OUTPUT_REPORT  = 'statistical_validation_report.txt'
OUTPUT_FIG     = 'bootstrap_confidence_intervals.png'
rng = np.random.RandomState(SEED)


# ------ HELPERS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """
    Compute bootstrap 95% CI for the mean of `values`.
    Returns (mean, ci_lower, ci_upper, std).
    """
    values = np.array(values, dtype=float)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    ci_lo = np.percentile(boot_means, 100 * alpha / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), float(ci_lo), float(ci_hi), float(np.std(values))


def rank_biserial(x, y):
    """
    Effect size for Wilcoxon signed-rank test.
    Rank-biserial r = 1 - 2W / (n(n+1)/2)
    """
    n = len(x)
    diffs = np.array(x, dtype=float) - np.array(y, dtype=float)
    diffs = diffs[diffs != 0]   # exclude ties
    n_eff = len(diffs)
    if n_eff == 0:
        return 0.0
    res = stats.wilcoxon(x, y)
    W = res.statistic
    r = 1 - (2 * W) / (n_eff * (n_eff + 1) / 2)
    return float(r)


def wilcoxon_test(a, b, alternative):
    """
    Wilcoxon signed-rank test between paired arrays a and b.
    alternative: 'less' | 'greater'
    Returns dict with W, p, r (effect size), interpretation.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    try:
        W, p = stats.wilcoxon(a, b, alternative=alternative)
    except ValueError as e:
        return {'W': None, 'p': None, 'r': None, 'error': str(e)}
    r = rank_biserial(a, b)
    sig = p < ALPHA
    return {
        'W': float(W), 'p': float(p), 'r': float(r),
        'significant': bool(sig),
        'interpretation': (
            f"p={p:.4f} {'< ' if sig else '>= '}{ALPHA} --- "
            f"{'SIGNIFICANT' if sig else 'NOT significant'} | "
            f"effect size r={r:.3f}"
        )
    }


# ------ LOAD DATA ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("-" * 65)
print("STATISTICAL VALIDATION - Bootstrap CIs & Wilcoxon Tests")
print("-" * 65)

with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

individual = data.get('individual_results', [])
if len(individual) < 5:
    print(f"WARNING: Only {len(individual)} images found. "
          f"Wilcoxon is unreliable with fewer than 10 paired samples.\n"
          f"Run run_fidelity.py on more images first.")

lime_del  = [r['lime_deletion_auc']     for r in individual]
rand_del  = [r['random_deletion_auc']   for r in individual]
lime_ins  = [r['lime_insertion_auc']    for r in individual]
rand_ins  = [r['random_insertion_auc']  for r in individual]

n = len(lime_del)
print(f"\nN images = {n}")


# ------ BOOTSTRAP CIs ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("\n------ Bootstrap 95% Confidence Intervals ---------------------------------------------------------------------------")

ci_results = {}
metrics = {
    'LIME Deletion AUC':   lime_del,
    'Random Deletion AUC': rand_del,
    'LIME Insertion AUC':  lime_ins,
    'Random Insertion AUC':rand_ins,
}
for name, vals in metrics.items():
    mean, lo, hi, sd = bootstrap_ci(vals)
    ci_results[name] = {'mean': mean, 'ci_lower': lo, 'ci_upper': hi, 'std': sd}
    print(f"  {name:<25}: {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  SD={sd:.4f}")


# ------ WILCOXON TESTS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("\n------ Wilcoxon Signed-Rank Tests ------------------------------------------------------------------------------------------------------")

del_test = wilcoxon_test(lime_del, rand_del, alternative='less')
ins_test = wilcoxon_test(lime_ins, rand_ins, alternative='greater')

print(f"\n  Deletion  (H1: LIME < Random):  {del_test.get('interpretation', del_test.get('error'))}")
print(f"  Insertion (H1: LIME > Random):  {ins_test.get('interpretation', ins_test.get('error'))}")


# ------ IMPROVEMENT DELTAS ------------------------------------------------------------------------------------------------------------------------------------------------------------------

del_deltas = [r - l for l, r in zip(lime_del, rand_del)]  # positive = LIME better
ins_deltas = [l - r for l, r in zip(lime_ins, rand_ins)]  # positive = LIME better

del_delta_mean, del_delta_lo, del_delta_hi, _ = bootstrap_ci(del_deltas)
ins_delta_mean, ins_delta_lo, ins_delta_hi, _ = bootstrap_ci(ins_deltas)

print(f"\n------ LIME Improvement Over Random ------------------------------------------------------------------------------------------------")
print(f"  Deletion  improvement: {del_delta_mean:+.4f}  95% CI [{del_delta_lo:+.4f}, {del_delta_hi:+.4f}]")
print(f"  Insertion improvement: {ins_delta_mean:+.4f}  95% CI [{ins_delta_lo:+.4f}, {ins_delta_hi:+.4f}]")


# ------ PLOT ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bootstrap 95% Confidence Intervals: LIME vs Random Baseline', fontsize=14, fontweight='bold')

labels    = ['LIME', 'Random']
del_means = [ci_results['LIME Deletion AUC']['mean'],   ci_results['Random Deletion AUC']['mean']]
del_lo    = [ci_results['LIME Deletion AUC']['ci_lower'], ci_results['Random Deletion AUC']['ci_lower']]
del_hi    = [ci_results['LIME Deletion AUC']['ci_upper'], ci_results['Random Deletion AUC']['ci_upper']]

ins_means = [ci_results['LIME Insertion AUC']['mean'],   ci_results['Random Insertion AUC']['mean']]
ins_lo    = [ci_results['LIME Insertion AUC']['ci_lower'], ci_results['Random Insertion AUC']['ci_lower']]
ins_hi    = [ci_results['LIME Insertion AUC']['ci_upper'], ci_results['Random Insertion AUC']['ci_upper']]

colors = ['steelblue', 'lightcoral']

for ax, means, lo, hi, title, ylabel in [
    (axes[0], del_means, del_lo, del_hi, 'Deletion AUC (Lower = Better)', 'AUC'),
    (axes[1], ins_means, ins_lo, ins_hi, 'Insertion AUC (Higher = Better)', 'AUC'),
]:
    yerr_lo = [m - l for m, l in zip(means, lo)]
    yerr_hi = [h - m for m, h in zip(means, hi)]
    bars = ax.bar(labels, means, color=colors, alpha=0.85,
                  yerr=[yerr_lo, yerr_hi], capsize=8, error_kw={'linewidth': 2})
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches='tight')
print(f"\n--- Figure saved: {OUTPUT_FIG}")


# ------ SAVE RESULTS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

output = {
    'n_images': n,
    'bootstrap_ci': ci_results,
    'wilcoxon_deletion':  del_test,
    'wilcoxon_insertion': ins_test,
    'lime_improvement': {
        'deletion':  {'mean': del_delta_mean, 'ci_lower': del_delta_lo, 'ci_upper': del_delta_hi},
        'insertion': {'mean': ins_delta_mean, 'ci_lower': ins_delta_lo, 'ci_upper': ins_delta_hi},
    }
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)
print(f"--- Results saved: {OUTPUT_JSON}")

report = f"""
STATISTICAL VALIDATION REPORT
==============================
N images evaluated: {n}
Bootstrap iterations: {N_BOOTSTRAP}
Significance level: alpha = {ALPHA}

BOOTSTRAP CONFIDENCE INTERVALS (95%)
--------------------------------------
LIME Deletion AUC:    {ci_results['LIME Deletion AUC']['mean']:.4f}  [{ci_results['LIME Deletion AUC']['ci_lower']:.4f}, {ci_results['LIME Deletion AUC']['ci_upper']:.4f}]
Random Deletion AUC:  {ci_results['Random Deletion AUC']['mean']:.4f}  [{ci_results['Random Deletion AUC']['ci_lower']:.4f}, {ci_results['Random Deletion AUC']['ci_upper']:.4f}]
LIME Insertion AUC:   {ci_results['LIME Insertion AUC']['mean']:.4f}  [{ci_results['LIME Insertion AUC']['ci_lower']:.4f}, {ci_results['LIME Insertion AUC']['ci_upper']:.4f}]
Random Insertion AUC: {ci_results['Random Insertion AUC']['mean']:.4f}  [{ci_results['Random Insertion AUC']['ci_lower']:.4f}, {ci_results['Random Insertion AUC']['ci_upper']:.4f}]

WILCOXON SIGNED-RANK TESTS
--------------------------------------
Deletion  (H1: LIME AUC < Random AUC):
  W={del_test.get('W')}, p={del_test.get('p'):.4f}, r={del_test.get('r'):.3f}
  {del_test.get('interpretation','')}

Insertion (H1: LIME AUC > Random AUC):
  W={ins_test.get('W')}, p={ins_test.get('p'):.4f}, r={ins_test.get('r'):.3f}
  {ins_test.get('interpretation','')}

LIME IMPROVEMENT OVER RANDOM
--------------------------------------
Deletion  improvement: {del_delta_mean:+.4f}  95% CI [{del_delta_lo:+.4f}, {del_delta_hi:+.4f}]
Insertion improvement: {ins_delta_mean:+.4f}  95% CI [{ins_delta_lo:+.4f}, {ins_delta_hi:+.4f}]
"""
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"--- Report saved: {OUTPUT_REPORT}")
print("\n" + report)
