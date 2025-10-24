import numpy as np
import pandas as pd
import json
import os

intv_mode = 'both'  # modality to intervene in: None, audio, visual or both'
training_data = None  # which training data was used (modality excluded)
complex = False  # whether the test data is complex or not

subset = 'success'

if training_data is not None and intv_mode is None: 
    failed_filename = f'alt_model/{training_data}_failed_questions.csv'
    new_filename = f'alt_model/{training_data}_success_questions'

elif training_data is not None and intv_mode is not None: 
    failed_filename = f'alt_model/{training_data}_{intv_mode}_failed_questions.csv'
    new_filename = f'alt_model/{training_data}_{intv_mode}_success_questions'

elif intv_mode is not None:
    failed_filename = f'intv/{intv_mode}_failed_questions.csv'
    new_filename = f'intv/{intv_mode}_success_questions'

else:
    failed_filename = 'failed_questions.csv'
    new_filename = 'success_questions'

org_json = json.load(open('../../json/avqa-test.json', 'r'))


if complex:
    failed_filename = 'complex/' + failed_filename
    new_filename = 'complex/' + new_filename
    org_json = json.load(open('../../json/avqa-test-complex.json', 'r'))

elif subset is not None:
    failed_filename = f'{subset}/{intv_mode}_failed_questions.csv'
    new_filename = f'{subset}/{intv_mode}_success_questions'
    org_json = json.load(open(f'../../json/avqa-test-{subset}.json', 'r'))

print(f'Loading failed questions from {failed_filename}')

df = pd.read_csv(failed_filename)
failed_idx = df['idx'].tolist()

print(f'Number of failed questions: {len(failed_idx)}')

print(len(org_json))

for idx in reversed(failed_idx):
    org_json.pop(idx)



with open(f'{new_filename}.json', 'w') as f:
    json.dump(org_json, f, indent = 4)

print(f'Cleaned json saved to {new_filename}.json')

check_json = json.load(open(f'{new_filename}.json', 'r'))
print(len(check_json))