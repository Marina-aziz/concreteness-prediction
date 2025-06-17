import pandas as pd
import matplotlib.pyplot as plt

# Load  data
arabic = pd.read_excel('200 arabic words.xlsx')
english = pd.read_excel('english_nouns_all.xlsx')

arabic_mean_col = 'Arabic mean'
arabic_sd_col = 'Arabic sd'
english_mean_col = 'Concreteness'
english_sd_col = 'SD'

col_en = '#1f77b4'    # muted blue
col_ar = '#ff7f0e'    # orange

#  ARABIC COMBINED PLOT 
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})

# Left: Croissant plot
axes[0].scatter(arabic[arabic_mean_col], arabic[arabic_sd_col], alpha=0.7, color=col_ar, edgecolor='k', s=30)
axes[0].set_xlabel('Mean Concreteness Rating')
axes[0].set_ylabel('Standard Deviation')
axes[0].set_title('Arabic: Mean concreteness vs. SD (Scatter)')
axes[0].set_xlim(1.0, 5.0)

# Right: Violin plot
axes[1].violinplot(arabic[arabic_mean_col].dropna(), vert=True, showmeans=True)
for body in axes[1].collections:
    body.set_facecolor(col_ar)
    body.set_edgecolor('black')
axes[1].set_ylabel('Mean Concreteness Rating')
axes[1].set_xticks([])
axes[1].set_title('Distribution (Violin)')
axes[1].set_ylim(0.0, 6.0)
axes[1].set_yticks([0, 1, 2, 3, 4, 5, 6])
plt.suptitle('Arabic Dataset: Rating Agreement and Distribution')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('arabic_agreement_and_distribution.pdf')
plt.close()

#  ENGLISH COMBINED PLOT 
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})

# Left: Croissant plot
axes[0].scatter(english[english_mean_col], english[english_sd_col], alpha=0.7, color=col_en, edgecolor='k', s=30)
axes[0].set_xlabel('Mean Concreteness Rating')
axes[0].set_ylabel('Standard Deviation')
axes[0].set_title('English: Mean concreteness vs. SD (Scatter)')

# Right: Violin plot
axes[1].violinplot(english[english_mean_col].dropna(), vert=True, showmeans=True)
for body in axes[1].collections:
    body.set_facecolor(col_en)
    body.set_edgecolor('black')
axes[1].set_ylabel('Mean Concreteness Rating')
axes[1].set_xticks([])
axes[1].set_ylim(0.0, 6.0)
axes[1].set_yticks([1, 2, 3, 4, 5])
axes[1].set_title('Distribution (Violin)')
plt.suptitle('English Dataset: Rating Agreement and Distribution')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('english_agreement_and_distribution.pdf')
plt.close()
