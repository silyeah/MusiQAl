import numpy as np
import pandas as pd
import json
import os

failed_filename = '../net_grd_avst/test_results/failed_questions.csv'

df = pd.read_csv(failed_filename)
failed_idx = df['idx'].tolist()

org_json = json.load(open('avqa-test.json', 'r'))

print(len(org_json))


for idx in reversed(failed_idx):
    org_json.pop(idx)

print(len(org_json))


with open('avqa-test-success-lr.json', 'w') as f:
    json.dump(org_json, f, indent = 4)

print('Cleaned json saved to avqa-test-success-lr.json')

check_json = json.load(open('avqa-test-success-lr.json', 'r'))
print(len(check_json))

print(check_json[0])
