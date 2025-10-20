import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data (you can replace these with your real values)
data = {
    'Category': ['Causal', 'Comparative', 'Counting', 'Existential', 'Location', 'Purpose', 'Temporal'],
    'Audio': [30, 550, 700, 800, 700, 0, 150],
    'Audio-Visual': [470, 650, 1800, 1700, 450, 320, 950],
    'Visual': [280, 0, 1500, 100, 900, 0, 600]
}

# Convert to long-form DataFrame for seaborn
df = pd.DataFrame(data)
df_melt = df.melt(id_vars='Category', var_name='Modality', value_name='Counts')

# Set global font sizes
plt.rcParams.update({
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # X and Y axis label font size
    'xtick.labelsize': 13,     # X tick labels
    'ytick.labelsize': 13,     # Y tick labels
    'legend.fontsize': 13,     # Legend text
    'legend.title_fontsize': 14, # Legend title
    'figure.titlesize': 20     # Figure-level title (if used)
})

# Create plot
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_melt,
    x='Category',
    y='Counts',
    hue='Modality',
    palette=['#b896c6', '#b36b24', '#6aa88d']  # Customize colors if desired
)

# Rotate category labels and style the axes
plt.xticks(rotation=45, ha='right')
plt.xlabel('Category', fontsize=16)
plt.ylabel('Counts', fontsize=16)
plt.title('Counts by Category and Modality', fontsize=18, pad=15)

plt.legend(
    title='Modality excluded',
    title_fontsize=14,
    fontsize=13,
    bbox_to_anchor=(1.02, 1),  # moves legend to the right
    loc='upper left',
    borderaxespad=0
)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend


# Save as PNG (high resolution)
plt.savefig('test_fig.png', dpi=300, bbox_inches='tight')
