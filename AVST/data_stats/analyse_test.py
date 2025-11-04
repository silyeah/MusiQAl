import json
import os
import pandas as pd
import ast


category = 'test'  # 'train', 'val' or 'test'


samples = json.load(open(f"../json/avqa-{category}.json", 'r'))
stats = pd.read_csv(f"{category}_reduced.csv")

hallvard_stats = pd.read_csv('uniform_vs_testacc_values.csv')


A_ext = []
A_count = []
A_cmp = []
A_temp = []
A_caus = []
V_ext = []
V_loc = []
V_count = []
V_temp = []
V_caus = []
AV_ext = []
AV_count = []
AV_loc = []
AV_cmp = []
AV_temp = []
AV_caus = []
AV_purp = []



for x in samples:
    q_id = x["question_id"]
    row = stats[stats['question_id'] == q_id]
    no_answers = row['no_answers'].values[0]

    type =ast.literal_eval(x['type'])

    if type[0] == 'Audio':
        if type[1] == 'Existential':
            A_ext.append(1/no_answers)
        elif type[1] == 'Counting':
            A_count.append(1/no_answers)
        elif type[1] == 'Comparative':
            A_cmp.append(1/no_answers)
        elif type[1] == 'Temporal':
            A_temp.append(1/no_answers)
        elif type[1] == 'Causal':
            A_caus.append(1/no_answers)
    elif type[0] == 'Visual':
        if type[1] == 'Existential':
            V_ext.append(1/no_answers)
        elif type[1] == 'Location':
            V_loc.append(1/no_answers)
        elif type[1] == 'Counting':
            V_count.append(1/no_answers)
        elif type[1] == 'Temporal':
            V_temp.append(1/no_answers)
        elif type[1] == 'Causal':
            V_caus.append(1/no_answers)
    elif type[0] == 'Audio-Visual':
        if type[1] == 'Existential':
            AV_ext.append(1/no_answers)
        elif type[1] == 'Counting':
            AV_count.append(1/no_answers)
        elif type[1] == 'Location':
            AV_loc.append(1/no_answers)
        elif type[1] == 'Comparative':
            AV_cmp.append(1/no_answers)
        elif type[1] == 'Temporal':
            AV_temp.append(1/no_answers)
        elif type[1] == 'Causal':
            AV_caus.append(1/no_answers)
        elif type[1] == 'Purpose':
            AV_purp.append(1/no_answers)

    else: 
        print("Unknown type:", type)


testacc_pct = hallvard_stats['testacc_pct']/100

A_ext_pmax = testacc_pct[0]
A_count_pmax = testacc_pct[1]
A_cmp_pmax = testacc_pct[4]
A_temp_pmax = testacc_pct[2]
A_caus_pmax = testacc_pct[3]

V_ext_pmax = testacc_pct[12]
V_loc_pmax = testacc_pct[14]
V_count_pmax = testacc_pct[13]
V_temp_pmax = testacc_pct[15]
V_caus_pmax = testacc_pct[16]

AV_ext_pmax = testacc_pct[5]
AV_count_pmax = testacc_pct[6]
AV_loc_pmax = testacc_pct[7]
AV_cmp_pmax = testacc_pct[11]
AV_temp_pmax = testacc_pct[8]
AV_caus_pmax = testacc_pct[9]
AV_purp_pmax = testacc_pct[10]

weighted_audio_overall = (A_ext_pmax * len(A_ext) + A_count_pmax * len(A_count) + A_cmp_pmax * len(A_cmp) + A_temp_pmax * len(A_temp) + A_caus_pmax * len(A_caus))/ ((len(A_ext) + len(A_count) + len(A_cmp) + len(A_temp) + len(A_caus)))
weighted_video_overall = (V_ext_pmax * len(V_ext) + V_loc_pmax * len(V_loc) + V_count_pmax * len(V_count) + V_temp_pmax * len(V_temp) + V_caus_pmax * len(V_caus))/ ((len(V_ext) + len(V_loc) + len(V_count) + len(V_temp) + len(V_caus)))
weighted_av_overall = (AV_ext_pmax * len(AV_ext) + AV_count_pmax * len(AV_count) + AV_loc_pmax * len(AV_loc) + AV_cmp_pmax * len(AV_cmp) + AV_temp_pmax * len(AV_temp) + AV_caus_pmax * len(AV_caus) + AV_purp_pmax * len(AV_purp))/ ((len(AV_ext) + len(AV_count) + len(AV_loc) + len(AV_cmp) + len(AV_temp) + len(AV_caus) + len(AV_purp)))
weighted_total_overall = (weighted_audio_overall * (len(A_ext) + len(A_count) + len(A_cmp) + len(A_temp) + len(A_caus)) + weighted_video_overall * (len(V_ext) + len(V_loc) + len(V_count) + len(V_temp) + len(V_caus)) + weighted_av_overall * (len(AV_ext) + len(AV_count) + len(AV_loc) + len(AV_cmp) + len(AV_temp) + len(AV_caus) + len(AV_purp)))
stats_filename = f"question_stats_{category}.csv"

total_samples = len(A_ext)+len(A_count)+len(A_cmp)+len(A_temp)+len(A_caus)+len(V_ext)+len(V_loc)+len(V_count)+len(V_temp)+len(V_caus)+len(AV_count)+len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp)+len(AV_caus)+len(AV_purp)



with open(stats_filename, 'w') as f:
    f.write("question_category,prob_correct_uniform,prob_correct_freq,question_count\n")
    f.write(f'audio_existential,{sum(A_ext)/len(A_ext)},{A_ext_pmax},{len(A_ext)}\n')
    f.write(f'audio_counting,{sum(A_count)/len(A_count)},{A_count_pmax},{len(A_count)}\n')
    f.write(f'audio_comparison,{sum(A_cmp)/len(A_cmp)},{A_cmp_pmax},{len(A_cmp)}\n')
    f.write(f'audio_temporal,{sum(A_temp)/len(A_temp)},{A_temp_pmax},{len(A_temp)}\n')
    f.write(f'audio_causal,{sum(A_caus)/len(A_caus)},{A_caus_pmax},{len(A_caus)}\n')
    f.write(f'audio_overall,{(sum(A_ext)+sum(A_count)+sum(A_cmp)+sum(A_temp)+sum(A_caus)) / (len(A_ext)+len(A_count)+len(A_cmp)+len(A_temp)+len(A_caus))},{weighted_audio_overall},{(len(A_ext)+len(A_count)+len(A_cmp)+len(A_temp)+len(A_caus))}\n')

    f.write(f'visual_existential,{sum(V_ext)/len(V_ext)},{V_ext_pmax},{len(V_ext)}\n')
    f.write(f'visual_localization,{sum(V_loc)/len(V_loc)},{V_loc_pmax},{len(V_loc)}\n')
    f.write(f'visual_counting,{sum(V_count)/len(V_count)},{V_count_pmax},{len(V_count)}\n')
    f.write(f'visual_temporal,{sum(V_temp)/len(V_temp)},{V_temp_pmax},{len(V_temp)}\n')
    f.write(f'visual_causal,{sum(V_caus)/len(V_caus)},{V_caus_pmax},{len(V_caus)}\n')
    f.write(f'visual_overall,{(sum(V_ext)+sum(V_loc)+sum(V_count)+sum(V_temp)+sum(V_caus)) / (len(V_ext)+len(V_loc)+len(V_count)+len(V_temp)+len(V_caus))},{weighted_video_overall},{(len(V_ext)+len(V_loc)+len(V_count)+len(V_temp)+len(V_caus))}\n')

    f.write(f'av_existential,{sum(AV_ext)/len(AV_ext)},{AV_ext_pmax},{len(AV_ext)}\n')
    f.write(f'av_counting,{sum(AV_count)/len(AV_count)},{AV_count_pmax},{len(AV_count)}\n')
    f.write(f'av_localization,{sum(AV_loc)/len(AV_loc)},{AV_loc_pmax},{len(AV_loc)}\n')
    f.write(f'av_comparison,{sum(AV_cmp)/len(AV_cmp)},{AV_cmp_pmax},{len(AV_cmp)}\n')
    f.write(f'av_temporal,{sum(AV_temp)/len(AV_temp)},{AV_temp_pmax},{len(AV_temp)}\n')
    f.write(f'av_causal,{sum(AV_caus)/len(AV_caus)},{AV_caus_pmax},{len(AV_caus)}\n')
    f.write(f'av_purpose,{sum(AV_purp)/len(AV_purp)},{AV_purp_pmax},{len(AV_purp)}\n')
    f.write(f'av_overall,{(sum(AV_count)+sum(AV_loc)+sum(AV_ext)+sum(AV_temp)+sum(AV_cmp)+sum(AV_caus)+sum(AV_purp)) / (len(AV_count)+len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp)+len(AV_caus)+len(AV_purp))},{weighted_audio_overall},{(len(AV_count)+len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp)+len(AV_caus)+len(AV_purp))}\n')

    f.write(f'overall,{(sum(A_ext)+sum(A_count)+sum(A_cmp)+sum(A_temp)+sum(A_caus)+sum(V_ext)+sum(V_loc)+sum(V_count)+sum(V_temp)+sum(V_caus)+sum(AV_count)+sum(AV_loc)+sum(AV_ext)+sum(AV_temp)+sum(AV_cmp)+sum(AV_caus)+sum(AV_purp))/total_samples},{weighted_total_overall/total_samples},{total_samples}\n')

    


print(f"Saved stats to {stats_filename}")

print("Sum of elements check:", len(A_ext)+len(A_count)+len(A_cmp)+len(A_temp)+len(A_caus)+len(V_ext)+len(V_loc)+len(V_count)+len(V_temp)+len(V_caus)+len(AV_count)+len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp)+len(AV_caus)+len(AV_purp))
print("Total questions:", len(samples))

