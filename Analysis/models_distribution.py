import pandas as pd
import matplotlib.pyplot as plt

# Load data
arabic = pd.read_csv('ar_merged_predictions.csv')
english = pd.read_csv('en_merged_predictions.csv')

# Plot colors
knn_color   = "#dabfff"  # pastel purple
gpt_color   = "#907ad6"  # pastel blue 
human_color = "#2c2a4a"  # indigo

# Function to plot three violins side by side
def plot_three_violins(data, title, filename):
    # make the box larger
    fig, ax = plt.subplots(figsize=(7, 7))
    
    vp = ax.violinplot(
        [data['human'].dropna(), data['knn_pred'].dropna(), data['gpt_pred'].dropna()],
        showmeans=True,
        vert=True
    )
    for idx, body in enumerate(vp['bodies']):
        color = [human_color, knn_color, gpt_color][idx]
        body.set_facecolor(color)
        body.set_edgecolor('black')
    vp['cmeans'].set_color('black')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Human', 'KNN', 'GPT'])
    ax.set_ylabel('Mean Concreteness Rating')
    # extend axis so violins aren’t cut off
    ax.set_ylim(0.0, 6.0)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title(title)
    
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

# Generate the two figures
plot_three_violins(arabic,
                   title='Arabic: Distribution of Human, KNN & GPT Ratings',
                   filename='arabic_models_violin.pdf')

plot_three_violins(english,
                   title='English: Distribution of Human, KNN & GPT Ratings',
                   filename='english_models_violin.pdf')
