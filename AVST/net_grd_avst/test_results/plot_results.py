import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_fig(modality, complex=False, alt_model=False, success=False):
    # Set file paths and title based on modality and flags

    if modality == 'audio':
        title_ref = 'Audio questions'

    elif modality == 'visual':
        title_ref = 'Visual questions'

    elif modality == 'av':
        title_ref = 'Audio-Visual questions'


    if alt_model:
        title = title_ref
        standard = 'final_results.csv'
        audio_intv = 'alt_model/audio_audio_final_results.csv'
        visual_intv = 'alt_model/visual_visual_final_results.csv'
        both_intv = 'alt_model/both_both_final_results.csv'

    elif complex: 
        title_ref = 'complex ' + title_ref
        title = f'Accuracy on {title_ref} questions by\ncategory and intervention type'
        standard = 'complex/final_results.csv'
        audio_intv = 'complex/intv/audio_final_results.csv'
        visual_intv = 'complex/intv/visual_final_results.csv'
        both_intv = 'complex/intv/both_final_results.csv'

    elif success == 'success':
        title = f'Accuracy on {title_ref} questions by\ncategory and intervention type,\n where model succeeds without intervention'
        standard = 'final_results_success.csv'
        audio_intv = 'success/audio_final_results.csv'
        visual_intv = 'success/visual_final_results.csv'
        both_intv = 'success/both_final_results.csv'
    

    elif success == 'failure':
        title = title_ref
        standard = 'final_results_failure.csv'
        audio_intv = 'failure/audio_final_results.csv'
        visual_intv = 'failure/visual_final_results.csv'
        both_intv = 'failure/both_final_results.csv'

    else: 
        # File paths
        title = title_ref
        standard = 'final_results.csv'
        audio_intv = 'intv/audio_final_results.csv'
        visual_intv = 'intv/visual_final_results.csv'
        both_intv = 'intv/both_final_results.csv'


    # Category mapping based on modality
    if modality == 'audio':
        if complex:    
            category_list = ['Existential', 'Counting', 'Temporal', 'Causal', 'Total']
            df_categories = ['audio_existential_accuracy', 'audio_counting_accuracy',
                            'audio_temporal_accuracy', 'audio_causal_accuracy', 'audio_overall_accuracy']

        else:
            category_list = ['Existential', 'Counting', 'Comparative', 'Temporal', 'Causal', 'Total']
            df_categories = ['audio_existential_accuracy', 'audio_counting_accuracy', 'audio_comparison_accuracy',
                            'audio_temporal_accuracy', 'audio_causal_accuracy', 'audio_overall_accuracy']
        
        colours = ["firebrick", "goldenrod", "forestgreen", "slateblue"]

        if success == 'failure':
            colours = ["goldenrod", "forestgreen", "slateblue"]

    elif modality == 'visual':
        if complex:
            category_list = ['Location', 'Counting','Causal', 'Total']
            df_categories = ['visual_localization_accuracy', 'visual_counting_accuracy',
                            'visual_causal_accuracy', 'visual_overall_accuracy']
            colours = ["cornflowerblue", "darksalmon", "yellowgreen", "teal"]
        
        elif success == 'failure':
            category_list = ['Location', 'Counting', 'Temporal', 'Causal', 'Total']
            df_categories = ['visual_localization_accuracy', 'visual_counting_accuracy',
                            'visual_temporal_accuracy', 'visual_causal_accuracy', 'visual_overall_accuracy']
            colours = ["darksalmon", "yellowgreen", "teal"]

        else:
            category_list = ['Existential', 'Location', 'Counting', 'Temporal', 'Causal', 'Total']
            df_categories = ['visual_existential_accuracy', 'visual_localization_accuracy', 'visual_counting_accuracy',
                            'visual_temporal_accuracy', 'visual_causal_accuracy', 'visual_overall_accuracy']
            colours = ["cornflowerblue", "darksalmon", "yellowgreen", "teal"]

    elif modality == 'av':
        if complex:
            category_list = ['Existential', 'Counting', 'Location', 'Temporal', 'Causal', 'Purpose', 'Total']
            df_categories = ['av_existential_accuracy', 'av_counting_accuracy', 'av_localization_accuracy',
                            'av_temporal_accuracy', 'av_causal_accuracy',
                            'av_purpose_accuracy', 'av_overall_accuracy']
        else: 
            category_list = ['Existential', 'Counting', 'Location', 'Comparative', 'Temporal', 'Causal', 'Purpose', 'Total']
            df_categories = ['av_existential_accuracy', 'av_counting_accuracy', 'av_localization_accuracy',
                            'av_comparison_accuracy', 'av_temporal_accuracy', 'av_causal_accuracy',
                            'av_purpose_accuracy', 'av_overall_accuracy']
        colours = ['#b896c6', '#b36b24', '#6aa88d', '#c1a35f']

        if success == 'failure':
            colours = ['#b36b24', '#6aa88d', '#c1a35f']
        



    def get_accuracies(file_path, categories):
        df = pd.read_csv(file_path)
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce').fillna(0)
        
        result = []
        for cat in categories:
            value = df.loc[df['question_category'] == cat, 'accuracy'].values
            result.append(value[0] if len(value) else None)
        
        #print(result)
        return result


    # --- Load accuracies from all files ---
    standard_accuracies = get_accuracies(standard, df_categories)
    audio_intv_accuracies = get_accuracies(audio_intv, df_categories)
    visual_intv_accuracies = get_accuracies(visual_intv, df_categories)
    both_intv_accuracies = get_accuracies(both_intv, df_categories)

    # --- Build DataFrame for plotting ---
    data = {
        'Question category': category_list,
        'None': standard_accuracies,
        'Visual': visual_intv_accuracies,
        'Audio': audio_intv_accuracies,
        'Both': both_intv_accuracies
    }

    if success is not None:
        data = {
        'Question category': category_list,
        'Visual': visual_intv_accuracies,
        'Audio': audio_intv_accuracies,
        'Both': both_intv_accuracies
    }


    df = pd.DataFrame(data)

    # Convert to long format for seaborn
    df_melt = df.melt(id_vars='Question category', var_name='Modality excluded', value_name='Accuracy')
    
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
        hue='Modality excluded',
        palette=colours  #['#b896c6', '#b36b24', '#6aa88d', '#c1a35f']
    )

    # Axis formatting
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Question category', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.title(title, fontsize=24, pad=15)

    plt.legend(
        title='Modality\nexcluded',
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
    if alt_model:
        plt.savefig(f'figures/fig_{modality}_alt_model.png', dpi=300, bbox_inches='tight') 
    elif success is not None:
        plt.savefig(f'figures/fig_{modality}_{success}.png', dpi=300, bbox_inches='tight')
    elif complex:
        plt.savefig(f'figures/fig_{modality}_complex.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'figures/fig_{modality}.png', dpi=300, bbox_inches='tight')



complex = False
alt_model = False
success = None

make_fig('audio', complex=complex, alt_model=alt_model, success=success)
make_fig('visual', complex=complex, alt_model=alt_model, success=success)
make_fig('av', complex=complex, alt_model=alt_model, success=success)


# complex = True
# alt_model = False
# success = False

# make_fig('audio', complex=complex, alt_model=alt_model, success=success)
# make_fig('visual', complex=complex, alt_model=alt_model, success=success)
# make_fig('av', complex=complex, alt_model=alt_model, success=success)

# complex = False
# alt_model = True
# success = False

# make_fig('audio', complex=complex, alt_model=alt_model, success=success)
# make_fig('visual', complex=complex, alt_model=alt_model, success=success)
# make_fig('av', complex=complex, alt_model=alt_model, success=success)


complex = False
alt_model = False
success = 'failure'

make_fig('audio', complex=complex, alt_model=alt_model, success=success)
make_fig('visual', complex=complex, alt_model=alt_model, success=success)
make_fig('av', complex=complex, alt_model=alt_model, success=success)

