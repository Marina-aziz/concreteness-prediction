import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Load summary rows only
df = pd.read_csv("results_en_k_experiment.csv")
df = df[df["fold"] == "all"]          # keep only the 'all' row for each k
df = df.sort_values("k").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df["k"], df["spearman_corr"],
        marker="o", linewidth=1.4, label="English")

# x-axis: force 5-unit ticks 
ax.set_xlim(5, df["k"].max())
ax.xaxis.set_major_locator(MultipleLocator(5))

# mark best k with a star
best = df.loc[df["spearman_corr"].idxmax()]
ax.plot(best["k"], best["spearman_corr"],
        marker="*", markersize=14, color="#FFA07A")
ax.annotate(f"k={int(best['k'])}",
            (best["k"], best["spearman_corr"]),
            textcoords="offset points", xytext=(0, 6), ha="center")

ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("Spearman correlation")
ax.set_title("k-value tuning – English")

# y-axis: small head-room around the data
ax.set_ylim(df["spearman_corr"].min() - 0.01,
            df["spearman_corr"].max() + 0.03)

ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("k_tuning.pdf")
plt.close()
