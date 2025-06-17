import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load merged predictions (must have columns: human, knn_pred, gpt_pred)
df = pd.read_csv("merged_predictions.csv")

# Plot colors
knn_color   = "#dabfff"  # pastel purple
gpt_color   = "#907ad6"  # pastel blue 
ideal_color = "#2c2a4a"  # indigo

plt.figure(figsize=(6,6))

# K-NN scatter (tiny points)
plt.scatter(df['human'], df['knn_pred'],
            s=5, alpha=0.6, color=knn_color, label='K-NN')

# ChatGPT scatter (tiny points)
plt.scatter(df['human'], df['gpt_pred'],
            s=5, alpha=0.6, color=gpt_color, label='ChatGPT')

# Ideal y=x (thick dash)
plt.plot([1,5], [1,5],
         color=ideal_color,
         linewidth=3,
         linestyle='--',
         label='Ideal')

plt.xlabel('Human Mean Rating')
plt.ylabel('Model Prediction')
plt.title('Model vs. Human Concreteness Ratings')
plt.xlim(1,5); plt.ylim(1,5)
plt.xticks([1,2,3,4,5]); plt.yticks([1,2,3,4,5])
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig(
    "model_vs_human.pdf",
    format='pdf', dpi=600, bbox_inches='tight'
)
plt.close()