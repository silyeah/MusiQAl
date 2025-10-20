import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


complex = True

modality = 'visual'  # 'audio', 'visual', or 'av'

if modality == 'av':
    title_ref = 'audio-visual'
else:
    title_ref = modality

if complex: 
    title_ref = 'complex ' + title_ref
    standard = 'complex/final_results.csv'
    audio_intv = 'complex/intv/audio_final_results.csv'
    visual_intv = 'complex/intv/visual_final_results.csv'
    both_intv = 'complex/intv/both_final_results.csv'


else: 
    # File paths
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

elif modality == 'visual':
    if complex:
        category_list = ['Location', 'Counting','Causal', 'Total']
        df_categories = ['visual_localization_accuracy', 'visual_counting_accuracy',
                         'visual_causal_accuracy', 'visual_overall_accuracy']
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


# # --- Helper to extract accuracies ---
# def get_accuracies(file_path, categories):
#     df = pd.read_csv(file_path)
#     result = []
#     for cat in categories:
#         value = df.loc[df['question_category'] == cat, 'accuracy'].values
#         result.append(value[0] if len(value) else None)
    
#     print(result)
#     return result

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
    'Audio': audio_intv_accuracies,
    'Visual': visual_intv_accuracies,
    'Both': both_intv_accuracies
}

df = pd.DataFrame(data)

# Convert to long format for seaborn
df_melt = df.melt(id_vars='Question category', var_name='Modality excluded', value_name='Accuracy')

# --- Plot styling ---
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.title_fontsize': 14,
    'figure.titlesize': 20
})

# --- Create the bar plot ---
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_melt,
    x='Question category',
    y='Accuracy',
    hue='Modality excluded',
    palette=colours  #['#b896c6', '#b36b24', '#6aa88d', '#c1a35f']
)

# Axis formatting
plt.xticks(rotation=45, ha='right')
plt.xlabel('Question category', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.title(f'Accuracy on {title_ref} questions by\ncategory and intervention type', fontsize=18, pad=15)

plt.legend(
    title='Modality excluded',
    title_fontsize=14,
    fontsize=13,
    bbox_to_anchor=(1.02, 1),  # moves legend to the right
    loc='upper left',
    borderaxespad=0
)
plt.tight_layout(rect=[0, 0, 1, 1])  # leave space for legend

# --- Save as high-resolution PNG ---
if complex:
    plt.savefig(f'figures/fig_{modality}_complex.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(f'figures/fig_{modality}.png', dpi=300, bbox_inches='tight')
