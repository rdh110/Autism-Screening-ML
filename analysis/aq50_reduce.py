import pandas as pd
from shared_func import AQ10, get_models, get_outer_cv, get_selection_cv, nested_remove_questions

# Load the data
aq50_df = pd.read_csv("../data/cleaned_aq50.csv")

# drop label column
x = aq50_df.drop(columns=['autistic'])

# select label column
y = aq50_df['autistic']

# official AQ-10 columns in the dataset
aq10_columns = [f'aq_{i}' for i in AQ10]

# models to skip during analysis
skip_models = []

all_rows = []

# remove questions one at a time while using nested cross-validation
for name, model in get_models().items():
    if name in skip_models:
        continue

    print(f'\n=== {name} ===')
    rows = nested_remove_questions(model, x, y, get_outer_cv(), get_selection_cv(), 'AQ-10', aq10_columns)

    # label each row with model name
    for row in rows:
        row['Model'] = name

    # save this model's results to its own file
    file_name = name.lower().replace(' ', '_')
    pd.DataFrame(rows).to_csv(f'../results/aq50_reduce_{file_name}.csv', index=False)

    all_rows.extend(rows)

# save all models' results to a single file
results_df = pd.DataFrame(all_rows)
results_df.to_csv('../results/aq50_reduce_all_models.csv', index=False)


# average each mode's question sets over all test folds
average_results = results_df.groupby(['Model', 'Question Set Name'], sort=False)[['Accuracy', 'Precision', 'Recall', 'F1']].agg(['mean', 'std'])

average_results.columns = [f'{metric} {stat}' for metric, stat in average_results.columns]

# When bringing back for all models, remove {name} from file name
# save average results to a file
average_results.round(3).to_csv('../results/aq50_reduce_average_all_models.csv')
print(average_results.round(3))


