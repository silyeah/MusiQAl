import numpy as np
import pandas as pd
import json
import os


org_json = json.load(open('avqa-test.json', 'r'))

remove_answers = []

for idx, qa in enumerate(org_json):
    if qa['answer'] == 'yes' or qa['answer'] == 'no': #or qa['answer'] == 'increase' or qa['answer'] == 'decrease':
        remove_answers.append(idx)

remove_answers.sort()

for idx in reversed(remove_answers):
    org_json.pop(idx)

print(len(org_json))

with open('avqa-test-complex.json', 'w') as f:
    json.dump(org_json, f, indent = 4)

