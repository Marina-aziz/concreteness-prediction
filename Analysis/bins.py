import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def smart_annotate(ax, x, y, text, dx=8, dy=6):
    """
    Place text to the left of (x,y) unless that would run off the left border;
    otherwise place it to the right.  dx,dy are offsets in points.
    """
    # star position in display coords
    star_disp = ax.transData.transform((x, y))
    # right border of the axes in display coords
    x_right = ax.transData.transform((ax.get_xlim()[1], y))[0]

    put_left = (star_disp[0] + ax.figure.dpi * dx / 72.0) > x_right
    # if text would cross the right border, put it on the left
    offset = (-dx if put_left else dx, dy)
    ha = "right" if put_left else "left"

    ax.annotate(text,
                (x, y),
                xytext=offset,
                textcoords="offset points",
                ha=ha, va="bottom")

# Load and keep summary rows only  
df = pd.read_csv("results_en_bins_experiment.csv")      
df = df[df["fold"] == "all"]

df = df.sort_values("bins").reset_index(drop=True)


fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(df["bins"], df["spearman_corr"],
        marker="o", linewidth=1.4, label="English")


ax.set_xlim(0, df["bins"].max())          # frame from 0 to max bin
ax.xaxis.set_major_locator(MultipleLocator(5))

# mark best bin size with a star
best = df.iloc[df["spearman_corr"].idxmax()]
ax.plot(best["bins"], best["spearman_corr"],
        marker="*", markersize=14, color="#FFA07A")

# smart_annotate(ax,
               # best["bins"], best["spearman_corr"],
               # f"bins = {int(best['bins'])}")

ax.annotate(f"bins={int(best['bins'])}",
            (best["bins"], best["spearman_corr"]),
            textcoords="offset points", xytext=(0, 6), ha="center")

ax.set_xlabel("Number of bins")
ax.set_ylabel("Spearman correlation")
ax.set_title("Bin-size tuning – English")

# y-axis zoomed to the data range ±0.01
ax.set_ylim(df["spearman_corr"].min() - 0.01,
            df["spearman_corr"].max() + 0.01)

ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bins_tuning.pdf")
plt.close()
