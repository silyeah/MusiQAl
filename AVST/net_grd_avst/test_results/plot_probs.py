import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Configuration: adjust fonts, colors, and style here
# -------------------------------------------------
plot_style = {
    "title": {
        "text": "Distribution of Highest Predicted Probability\nfor Correct Predictions",
        "fontsize": 26,
        "color": "black"
    },
    "xlabel": {
        "text": "Highest Predicted Probability",
        "fontsize": 22,
        "color": "black"
    },
    "ylabel": {
        "text": "Density",
        "fontsize": 22,
        "color": "black"
    },
    "legend": {
        "fontsize": 18,
        "title": 'Modality removed (mean, std.)',
        "title_fontsize": 22,
        "labelcolor": "black"
    },
    "line_colors": {
        "None": "#1f77b4",             # blue
        "Video": "#2ca02c",  # green
        "Audio": "#ff7f0e",   # orange
        "Both": "#d62728"    # red
    },
    "background_color": "white",
    "font_family": "DejaVu Sans",  # any Matplotlib font
    "figsize": (12, 6)
}

# -------------------------------------------------
# Data loading and processing
# -------------------------------------------------
standard = 'prediction_probs.csv'
audio_intv = 'intv/audio_prediction_probs.csv'
visual_intv = 'intv/visual_prediction_probs.csv'
both_intv = 'intv/both_prediction_probs.csv'

def get_probs(file_path):
    df = pd.read_csv(file_path)
    correct_probs = df[df['correct'] == 1]['probability']
    incorrect_probs = df[df['correct'] == 0]['probability']
    return correct_probs, incorrect_probs

# Get probabilities
std_correct, std_incorrect = get_probs(standard)
audio_correct, audio_incorrect  = get_probs(audio_intv)
visual_correct, visual_incorrect = get_probs(visual_intv)
both_correct, both_incorrect = get_probs(both_intv)


# -------------------------------------------------
# Plotting
# -------------------------------------------------
sns.set(style="whitegrid", font=plot_style["font_family"])
plt.figure(figsize=plot_style["figsize"], facecolor=plot_style["background_color"])

sns.kdeplot(std_correct, label=f'None ({np.mean(std_correct):.2f}, {np.std(std_correct):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["None"])
sns.kdeplot(visual_correct, label=f'Video ({np.mean(visual_correct):.2f}, {np.std(visual_correct):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["Video"])
sns.kdeplot(audio_correct, label=f'Audio ({np.mean(audio_correct):.2f}, {np.std(audio_correct):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["Audio"])
sns.kdeplot(both_correct, label=f'Both ({np.mean(both_correct):.2f}, {np.std(both_correct):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["Both"])

plt.title(plot_style["title"]["text"],
          fontsize=plot_style["title"]["fontsize"],
          color=plot_style["title"]["color"])
plt.xlabel(plot_style["xlabel"]["text"],
           fontsize=plot_style["xlabel"]["fontsize"],
           color=plot_style["xlabel"]["color"])
plt.ylabel(plot_style["ylabel"]["text"],
           fontsize=plot_style["ylabel"]["fontsize"],
           color=plot_style["ylabel"]["color"])

plt.legend(fontsize=plot_style["legend"]["fontsize"],
           title=plot_style["legend"]["title"],
           title_fontsize=plot_style["legend"]["title_fontsize"],
           labelcolor=plot_style["legend"]["labelcolor"])

plt.xlim(0, 1)

plt.tight_layout()
plt.savefig('figures/correct_prediction_probs.png', dpi=300, facecolor=plot_style["background_color"])
plt.close()









# -------------------------------------------------
# Configuration: adjust fonts, colors, and style here
# -------------------------------------------------
plot_style = {
    "title": {
        "text": "Distribution of Highest Predicted Probability\nfor Incorrect Predictions",
        "fontsize": 26,
        "color": "black"
    },
    "xlabel": {
        "text": "Highest Predicted Probability",
        "fontsize": 24,
        "color": "black"
    },
    "ylabel": {
        "text": "Density",
        "fontsize": 24,
        "color": "black"
    },
    "legend": {
        "fontsize": 18,
        "title": 'Modality removed (mean, std.)',
        "title_fontsize": 22,
        "labelcolor": "black"
    },
    "line_colors": {
        "None": "#1f77b4",             # blue
        "Video": "#2ca02c",  # green
        "Audio": "#ff7f0e",   # orange
        "Both": "#d62728"    # red
    },
    "background_color": "white",
    "font_family": "DejaVu Sans",  # any Matplotlib font
    "figsize": (12, 6)
}

# -------------------------------------------------
# Data loading and processing
# -------------------------------------------------



# -------------------------------------------------
# Plotting
# -------------------------------------------------
sns.set(style="whitegrid", font=plot_style["font_family"])
plt.figure(figsize=plot_style["figsize"], facecolor=plot_style["background_color"])

sns.kdeplot(std_incorrect, label=f'None ({np.mean(std_incorrect):.2f}, {np.std(std_incorrect):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["None"])
sns.kdeplot(visual_incorrect, label=f'Video ({np.mean(visual_incorrect):.2f}, {np.std(visual_incorrect):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["Video"])
sns.kdeplot(audio_incorrect, label=f'Audio ({np.mean(audio_incorrect):.2f}, {np.std(audio_incorrect):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["Audio"])
sns.kdeplot(both_incorrect, label=f'Both ({np.mean(both_incorrect):.2f}, {np.std(both_incorrect):.2f})', fill=True, alpha=0.5,
            color=plot_style["line_colors"]["Both"])

plt.title(plot_style["title"]["text"],
          fontsize=plot_style["title"]["fontsize"],
          color=plot_style["title"]["color"])
plt.xlabel(plot_style["xlabel"]["text"],
           fontsize=plot_style["xlabel"]["fontsize"],
           color=plot_style["xlabel"]["color"])
plt.ylabel(plot_style["ylabel"]["text"],
           fontsize=plot_style["ylabel"]["fontsize"],
           color=plot_style["ylabel"]["color"])

plt.legend(fontsize=plot_style["legend"]["fontsize"],
           title=plot_style["legend"]["title"],
           title_fontsize=plot_style["legend"]["title_fontsize"],
           labelcolor=plot_style["legend"]["labelcolor"])

plt.xlim(0, 1)

plt.tight_layout()
plt.savefig('figures/incorrect_prediction_probs.png', dpi=300, facecolor=plot_style["background_color"])
plt.close()

