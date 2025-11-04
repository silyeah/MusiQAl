import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def make_fig(modality,
             csv_path='question_stats_test.csv',
             to_percent=False,
             save_path=None,
             figsize=(9, 6),
             font='DejaVu Sans'):

    # --- Title setup ---
    if modality == 'av':
        title_ref = 'audio-visual'
    else:
        title_ref = modality

    title = f'Probability of success on {title_ref}\ntest questions by random guessing'

    # --- Category mapping based on modality ---
    if modality == 'audio':
        category_list = ['Existential', 'Counting', 'Comparative', 'Temporal', 'Causal', 'Total']
        df_categories = ['audio_existential', 'audio_counting', 'audio_comparison',
                         'audio_temporal', 'audio_causal', 'audio_overall']

        base_colours = ["firebrick"]

    elif modality == 'visual':
        category_list = ['Existential', 'Location', 'Counting', 'Temporal', 'Causal', 'Total']
        df_categories = ['visual_existential', 'visual_localization', 'visual_counting',
                         'visual_temporal', 'visual_causal', 'visual_overall']

        base_colours = ["teal"]

    elif modality == 'av':
        category_list = ['Existential', 'Counting', 'Location', 'Comparative', 'Temporal', 'Causal', 'Purpose',
                         'Total']
        df_categories = ['av_existential', 'av_counting', 'av_localization',
                         'av_comparison', 'av_temporal', 'av_causal',
                         'av_purpose', 'av_overall']

        base_colours = ['#b896c6']

    else:
        raise ValueError("modality must be 'audio', 'visual' or 'av'")

    # --- Helper: load CSV and extract values by category key ---
    def get_accuracies(file_path, categories):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Allow multiple possible prob column names:
        prob_col_candidates = ['prob_correct', 'probability', 'prob', 'accuracy', 'value']
        prob_col = None
        for c in prob_col_candidates:
            if c in df.columns:
                prob_col = c
                break
        if prob_col is None:
            raise KeyError(f"None of expected probability columns found in {file_path}. "
                           f"Looked for {prob_col_candidates}, got columns: {list(df.columns)}")

        # Ensure question category column exists
        if 'question_category' not in df.columns:
            raise KeyError(f"'question_category' column not found in {file_path}")

        # convert to numeric (NaN -> 0)
        df[prob_col] = pd.to_numeric(df[prob_col], errors='coerce').fillna(0)

        # collect values (in order of categories list)
        result = []
        for cat in categories:
            v = df.loc[df['question_category'] == cat, prob_col].values
            result.append(float(v[0]) if len(v) else None)
        return result

    # --- Load probabilities ---
    test_probs = get_accuracies(csv_path, df_categories)

    test_probs = [100.0*x for x in test_probs]
    y_label = 'Probability (%)'

    # --- Build DataFrame for plotting ---
    data = {
        'Question category': category_list,
        'Test data': test_probs,
    }
    df = pd.DataFrame(data)

    # No need to melt when there's only one value column; seaborn can plot from df directly.
    # But keep consistent ordering:
    order = category_list

    # Ensure palette has one color per category. If base_colours shorter, expand using seaborn palette.
    if len(base_colours) >= len(category_list):
        palette = base_colours[:len(category_list)]
    else:
        # create an expanded palette using the base colors as a palette seed
        palette = sns.color_palette(base_colours, n_colors=len(category_list))

    # --- Theme ---
    sns.set_theme(context='notebook', style='white', font=font,
                  rc={'axes.titlesize': 26, 'axes.labelsize': 20, 'xtick.labelsize': 16,
                      'ytick.labelsize': 16, 'legend.fontsize': 16, 'legend.title_fontsize': 18,
                      'figure.titlesize': 26})

    # --- Create the bar plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # plot bars: supply colors per bar
    #colors_for_bars = palette
    sns.barplot(
    data=df,
    x='Question category',
    y='Test data',
    hue='Question category',  # hue now required
    palette=palette,
    legend=False,             # hide redundant legend
    ax=ax
)

    # Axis formatting
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Question category', fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title(title, fontsize=26, pad=15)

    # Adjust tick font sizes explicitly
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)


    plt.tight_layout(rect=[0, 0, 1, 1])  # leave space for legend


    # --- Save as high-resolution PNG ---
    
    plt.savefig(f'data_probs_{modality}.png', dpi=300, bbox_inches='tight')



make_fig('audio')
make_fig('visual')
make_fig('av')
