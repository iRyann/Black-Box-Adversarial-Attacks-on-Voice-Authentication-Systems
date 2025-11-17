from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Config
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 11

# Load data
asv_adv = pd.read_csv("data/results/results_asv_adv.csv")
asv_sota = pd.read_csv("data/results/results_asv_sota.csv")
cm_adv = pd.read_csv("data/results/results_aasist3_adv.csv")
cm_sota = pd.read_csv("data/results/results_aasist3_sota.csv")

asv_adv["condition"] = "ADV"
asv_sota["condition"] = "SOTA"
cm_adv["condition"] = "ADV"
cm_sota["condition"] = "SOTA"

asv_all = pd.concat([asv_adv, asv_sota], ignore_index=True)
cm_all = pd.concat([cm_adv, cm_sota], ignore_index=True)

# Compute metrics
asv_stats = (
    asv_all.groupby("condition")
    .agg({"score": ["mean", "std", "count"], "prediction": "mean"})
    .round(3)
)

cm_all["detected"] = (cm_all["prediction"] == "Spoof").astype(int)
cm_stats = (
    cm_all.groupby("condition")
    .agg({"confidence": ["mean", "std"], "detected": "mean"})
    .round(3)
)
cm_adv["detected"] = (cm_adv["prediction"] == "Spoof").astype(int)
cm_sota["detected"] = (cm_sota["prediction"] == "Spoof").astype(int)

print("\n" + "=" * 60)
print("ASV METRICS (ECAPA-TDNN)")
print("=" * 60)
print(asv_stats)
print(f"\nAcceptance Rate:")
print(f"  ADV:  {asv_adv['prediction'].mean():.1%}")
print(f"  SOTA: {asv_sota['prediction'].mean():.1%}")

print("\n" + "=" * 60)
print("CM METRICS (AASIST)")
print("=" * 60)
print(cm_stats)
print(f"\nDetection Rate:")
print(f"  ADV:  {cm_adv['detected'].mean():.1%}")
print(f"  SOTA: {cm_sota['detected'].mean():.1%}")

# Create figures
Path("results/figures").mkdir(parents=True, exist_ok=True)

# Figure 1: ASV Scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    data=asv_all, x="condition", y="score", ax=ax1, palette=["#fc8d62", "#66c2a5"]
)
ax1.axhline(y=0.25, color="red", linestyle="--", linewidth=1.5, label="Threshold=0.25")
ax1.set_ylabel("Similarity Score", fontweight="bold")
ax1.set_xlabel("Condition", fontweight="bold")
ax1.set_title("ASV Verification Scores", fontweight="bold", fontsize=13)
ax1.legend()
ax1.set_ylim(-0.05, 1.05)

accept_rates = asv_all.groupby("condition")["prediction"].mean()
bars = ax2.bar(
    accept_rates.index, accept_rates.values, color=["#fc8d62", "#66c2a5"], alpha=0.8
)
ax2.set_ylabel("Acceptance Rate", fontweight="bold")
ax2.set_xlabel("Condition", fontweight="bold")
ax2.set_title("Attack Success Rate", fontweight="bold", fontsize=13)
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
for bar, val in zip(bars, accept_rates.values):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.03,
        f"{val:.1%}",
        ha="center",
        fontweight="bold",
        fontsize=12,
    )

sns.despine()
plt.tight_layout()
plt.savefig("results/figures/asv_comparison.png", bbox_inches="tight")
print("\n✅ Saved: results/figures/asv_comparison.png")

# Figure 2: CM Detection
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.violinplot(
    data=cm_all, x="condition", y="confidence", ax=ax1, palette=["#fc8d62", "#66c2a5"]
)
ax1.set_ylabel("CM Confidence (Bonafide)", fontweight="bold")
ax1.set_xlabel("Condition", fontweight="bold")
ax1.set_title("CM Confidence Distribution", fontweight="bold", fontsize=13)
ax1.set_ylim(0, 1.05)

detect_rates = cm_all.groupby("condition")["detected"].mean()
bars = ax2.bar(
    detect_rates.index, detect_rates.values, color=["#fc8d62", "#66c2a5"], alpha=0.8
)
ax2.set_ylabel("Detection Rate", fontweight="bold")
ax2.set_xlabel("Condition", fontweight="bold")
ax2.set_title("CM Spoof Detection", fontweight="bold", fontsize=13)
ax2.set_ylim(0, 1)
for bar, val in zip(bars, detect_rates.values):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.03,
        f"{val:.1%}",
        ha="center",
        fontweight="bold",
        fontsize=12,
    )

sns.despine()
plt.tight_layout()
plt.savefig("results/figures/cm_comparison.png", bbox_inches="tight")
print("✅ Saved: results/figures/cm_comparison.png")

# Figure 3: Combined summary
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics = pd.DataFrame(
    {
        "ADV": [asv_adv["prediction"].mean(), 1 - cm_adv["detected"].mean()],
        "SOTA": [asv_sota["prediction"].mean(), 1 - cm_sota["detected"].mean()],
    },
    index=["ASV Bypass", "CM Bypass"],
)

metrics.plot(kind="bar", ax=ax, color=["#fc8d62", "#66c2a5"], alpha=0.8, width=0.7)
ax.set_ylabel("Success Rate", fontweight="bold", fontsize=12)
ax.set_xlabel("Attack Type", fontweight="bold", fontsize=12)
ax.set_title("Overall Attack Performance: ADV vs SOTA", fontweight="bold", fontsize=14)
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
ax.legend(title="Condition", frameon=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

for container in ax.containers:
    ax.bar_label(
        container, fmt="%.1%%", label_type="edge", fontweight="bold", padding=3
    )

sns.despine()
plt.tight_layout()
plt.savefig("results/figures/overall_summary.png", bbox_inches="tight")
print("✅ Saved: results/figures/overall_summary.png")

# Summary table
summary = pd.DataFrame(
    {
        "Condition": ["ADV", "SOTA"],
        "N_samples": [len(asv_adv), len(asv_sota)],
        "ASV_score_mean": [asv_adv["score"].mean(), asv_sota["score"].mean()],
        "ASV_accept_rate": [
            asv_adv["prediction"].mean(),
            asv_sota["prediction"].mean(),
        ],
        "CM_confidence_mean": [
            cm_adv["confidence"].mean(),
            cm_sota["confidence"].mean(),
        ],
        "CM_detect_rate": [cm_adv["detected"].mean(), cm_sota["detected"].mean()],
    }
)
summary.to_csv("results/summary_table.csv", index=False)
print("✅ Saved: results/summary_table.csv")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
delta_asv = asv_sota["prediction"].mean() - asv_adv["prediction"].mean()
delta_cm = cm_sota["detected"].mean() - cm_adv["detected"].mean()

print(f"\nΔ ASV acceptance: {delta_asv:+.1%} (SOTA vs ADV)")
print(f"Δ CM detection:   {delta_cm:+.1%} (SOTA vs ADV)")

if delta_asv > 0.15:
    print("\n→ SOTA significantly MORE successful than ADV")
    print("  Transformations F1-F7 are OBSOLETE")
elif delta_asv < -0.15:
    print("\n→ ADV significantly MORE successful than SOTA")
    print("  Transformations F1-F7 remain USEFUL")
else:
    print("\n→ No significant difference between ADV and SOTA")
    print("  Transformations F1-F7 have MARGINAL impact")
