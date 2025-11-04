import pandas as pd
import json
import os


def split_excel_to_csv():
    excel_file = "all_questions_fixed.xlsx"

    df = pd.read_excel(excel_file)

    df.to_csv("all_questions.csv", index=False)

    first_col = df.columns[0]

    for value in df[first_col].unique():
        subset = df[df[first_col] == value]
        filename = f"{value}.csv"
        subset.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(subset)} rows")

        #Save only the third and last columns to a new CSV
        new_df = pd.read_csv(f"{value}.csv")
        new_df = new_df.iloc[:, [2, -1]]

        #rename second column name to 'no_answers'
        new_df.columns = [new_df.columns[0], 'no_answers']
        new_filename = f"{value}_reduced.csv"
        new_df.to_csv(new_filename, index=False)

    

def overall_chance():
    df_test = pd.read_csv('test.csv', skiprows=1, header=None)
    prob_sum_test = 0
    last_col_test = df_test.iloc[:, -1].tolist()
    for i in last_col_test:
        prob_sum_test += 1 / i
    
    overall_chance_test = prob_sum_test / len(df_test)
    print(f'Overall test chance: {overall_chance_test}')

    df_train = pd.read_csv('train.csv', skiprows=1, header=None)
    prob_sum_train = 0
    last_col_train = df_train.iloc[:, -1].tolist()
    for i in last_col_train:
        prob_sum_train += 1 / i

    overall_chance_train = prob_sum_train / len(df_train)
    print(f'Overall train chance: {overall_chance_train}')

    df_val = pd.read_csv('val.csv', skiprows=1, header=None)
    prob_sum_val = 0
    last_col_val = df_val.iloc[:, -1].tolist()
    for i in last_col_val:
        prob_sum_val += 1 / i

    overall_chance_val = prob_sum_val / len(df_val)
    print(f'Overall val chance: {overall_chance_val}')

    filename = "overall_chance_stats.csv"
    with open(filename, 'w') as f:
        f.write("split,overall_chance,question_count\n")
        f.write(f'test,{overall_chance_test},{len(df_test)}\n')
        f.write(f'train,{overall_chance_train},{len(df_train)}\n')
        f.write(f'val,{overall_chance_val},{len(df_val)}\n')



split_excel_to_csv()
overall_chance()

