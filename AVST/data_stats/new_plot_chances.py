import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_fig(modality):
    # Set file paths and title based on modality and flags
    file_path = 'question_stats_test.csv'

    if modality == 'audio':
        title = 'Audio questions'

    elif modality == 'visual':
        title = 'Visual questions'

    elif modality == 'av':
        title = 'Audio-Visual questions'


    # --- Category mapping based on modality ---
    if modality == 'audio':
        category_list = ['Existential', 'Counting', 'Comparative', 'Temporal', 'Causal', 'Total']
        df_categories = ['audio_existential', 'audio_counting', 'audio_comparison',
                         'audio_temporal', 'audio_causal', 'audio_overall']

        colours = ["lightcoral", "indianred"]

    elif modality == 'visual':
        category_list = ['Existential', 'Location', 'Counting', 'Temporal', 'Causal', 'Total']
        df_categories = ['visual_existential', 'visual_localization', 'visual_counting',
                         'visual_temporal', 'visual_causal', 'visual_overall']

        colours = ["lightseagreen", "mediumturquoise"]

    elif modality == 'av':
        category_list = ['Existential', 'Counting', 'Location', 'Comparative', 'Temporal', 'Causal', 'Purpose',
                         'Total']
        df_categories = ['av_existential', 'av_counting', 'av_localization',
                         'av_comparison', 'av_temporal', 'av_causal',
                         'av_purpose', 'av_overall']

        colours = ['#d4a6e0', '#9f7fb3']



    def get_accuracies(file_path, categories):
        df = pd.read_csv(file_path)
        df['prob_correct_uniform'] = pd.to_numeric(df['prob_correct_uniform'], errors='coerce').fillna(0)

        df['prob_correct_freq'] = pd.to_numeric(df['prob_correct_freq'], errors='coerce').fillna(0)
        
        result_uniform = []
        for cat in categories:
            value = df.loc[df['question_category'] == cat, 'prob_correct_uniform'].values*100
            result_uniform.append(value[0] if len(value) else None)

        results_freq = []

        for cat in categories:
            value = df.loc[df['question_category'] == cat, 'prob_correct_freq'].values*100
            results_freq.append(value[0] if len(value) else None)

        return result_uniform, results_freq


    # --- Load accuracies from all files ---
    uniform, freq = get_accuracies(file_path, df_categories)

    # --- Build DataFrame for plotting ---
    data = {
        'Question category': category_list,
        'Random': uniform,
        'Highest\nfrequency': freq,
    }


    df = pd.DataFrame(data)

    # Convert to long format for seaborn
    df_melt = df.melt(id_vars='Question category', var_name='Selection criterion', value_name='Accuracy')
    
    sns.set_theme(
        context='notebook',
        style='white',
        font='DejaVu Sans',
        rc={
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 14,
            'legend.title_fontsize': 18,
            'figure.titlesize': 20,
        }
    )

    # --- Create the bar plot ---
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=df_melt,
        x='Question category',
        y='Accuracy',
        hue='Selection criterion',
        palette=colours 
    )

    # Axis formatting
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Question category', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.title(title, fontsize=24, pad=15)

    plt.legend(
        title='Selection\ncriterion',
        title_fontsize=18,
        fontsize=16,
        bbox_to_anchor=(1.02, 1),  # moves legend to the right
        loc='upper left',
        borderaxespad=0
    )

    # 🔧 Adjust tick font sizes
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    ax = plt.gca()  # get current axes

    # Remove only the top and right spines (keep x and y axes visible)
    sns.despine(ax=ax, top=True, right=True)

    # Make the remaining spines (bottom and left) a bit lighter
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    # Remove legend frame (box around legend)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor('none')

    # Optionally, subtle horizontal gridlines (nice for bar plots)
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.grid(False, axis='x')



    plt.tight_layout(rect=[0, 0, 1, 1])  # leave space for legend


    # --- Save as high-resolution PNG ---
    plt.savefig(f'data_probs_{modality}.png', dpi=300, bbox_inches='tight')



make_fig('audio')
make_fig('visual')
make_fig('av')

    

