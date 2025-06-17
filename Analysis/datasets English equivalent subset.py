import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('200 arabic words.xlsx')

# Specify English column names
english_mean_col = 'English Mean'
english_sd_col = 'English sd'

df = df.drop_duplicates(subset='English word')

# Keep only complete rows
eng = df[[english_mean_col, english_sd_col]].dropna()

# Combined plot (scatter + violin)
fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                         gridspec_kw={'width_ratios': [3, 1]})

axes[0].scatter(eng[english_mean_col], eng[english_sd_col],
                alpha=0.7, edgecolor='k', s=30)
axes[0].set_xlabel('Mean Concreteness Rating')
axes[0].set_ylabel('Standard Deviation')
axes[0].set_title('English: Mean concreteness vs. SD (Scatter)')

axes[1].violinplot(eng[english_mean_col], vert=True, showmeans=True)
axes[1].set_ylabel('Mean Concreteness Rating')
axes[1].set_xticks([])
axes[1].set_title('Distribution (Violin)')

plt.suptitle('English Equivalent Subset: Rating Agreement and Distribution')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('en_subset_agreement_and_distribution.pdf')
plt.close()
