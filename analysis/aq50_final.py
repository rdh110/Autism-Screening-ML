import ast
import pandas as pd
from sklearn.model_selection import cross_validate
from shared_func import get_cv, get_models, AQ10

### Load the data ###
aq50_df = pd.read_csv('../data/cleaned_aq50.csv')

# drop label column
x = aq50_df.drop(columns=['autistic'])

# select label column
y = aq50_df['autistic']

# Official AQ-10 Questions (10 of the 50 questions from AQ-50)
aq10_columns = [f'aq_{i}' for i in AQ10]


### Pick the final short form: top 10 by frequency ###

# load question removal results (has the list of questions in every run)
all_df = pd.read_csv('../results/aq50_reduce_all_models.csv')

# keep the 25 runs (5 models x 5 folds) where 10 questions were left
runs = all_df[(all_df['Number of Questions'] == 10) &
              (all_df['Question Set Name'] != 'AQ-10')]

# count how many runs each question appeared in
counts = {}
for question_list in runs['Questions']:
    for question in ast.literal_eval(question_list):
        counts[question] = counts.get(question, 0) + 1

# sort most to least frequent and keep the top 10
frequency = pd.Series(counts).sort_values(ascending=False, kind='stable')
final_columns = list(frequency.index[:10])

print('Top 10 by frequency:')
print(frequency.head(10))

# warn if the 10th and 11th questions are tied, since the cut would be arbitrary
if len(frequency) > 10 and frequency.iloc[9] == frequency.iloc[10]:
    print('Warning: tie at 10th place, check which question to keep')


### Compare the final short form against the official AQ-10 ###
models = get_models()
results = []

question_sets = {
    'All 50': x,
    'Official AQ-10': x[aq10_columns],
    'Top 10 by Frequency': x[final_columns],
}

for set_name, x_set in question_sets.items():
    for name, model in models.items():
        print(f'\n{set_name} - {name}:')

        # same 25 train/test splits as the baseline in aq50_short.py
        scores = cross_validate(model, x_set, y, cv=get_cv(),
                                scoring=['accuracy', 'precision', 'recall', 'f1'])

        row = {'Model': name, 'Questions': set_name}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            test = scores[f'test_{metric}']
            print(f'{metric}: mean = {test.mean():.3f}, std = {test.std():.3f}')
            row[f'{metric.capitalize()} Mean'] = test.mean()
            row[f'{metric.capitalize()} Std'] = test.std()
        results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv('../results/aq50_final_vs_aq10.csv', index=False)

# side-by-side F1 table, one row per model
print('\nTest F1 by model:')
print(results_df.pivot(index='Model', columns='Questions', values='F1 Mean').round(3))
