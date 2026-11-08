import pandas as pd
import numpy as np
from plots import plot_pca
from sklearn.model_selection import cross_validate
from shared_func import SEED, get_cv, get_models, AQ10

aq50_df = pd.read_csv('../data/cleaned_aq50.csv')

# drop label column
x = aq50_df.drop(columns=['autistic'])

# select label column
y = aq50_df['autistic']

# Official AQ-10 Questions (10 of the 50 questions from AQ-50)
aq10_columns = [f'aq_{i}' for i in AQ10]
x_aq10 = x[aq10_columns]
print(x_aq10.shape)
print(x_aq10.columns.tolist())

# covariance matrix
cov_matrix = np.cov(x, rowvar=False)

### PCA portion ###

eigenvalues = np.linalg.eigvalsh(cov_matrix)[::-1]  # sort in descending order

# Share of total variance explained by each eigenvalue
variance_ratio = eigenvalues / eigenvalues.sum()

cumulative_variance_ratio = np.cumsum(variance_ratio)

print(f"First component explains {variance_ratio[0]:.1%}")
print(f"Components needed for 80%: {np.argmax(cumulative_variance_ratio >= 0.80) + 1}")
print(f"Components needed for 90%: {np.argmax(cumulative_variance_ratio >= 0.90) + 1}")

plot_pca(variance_ratio, 'AQ-50 PCA Variance explained', '../plots/aq50_pca_variance_explained.png')



### Cross Validation Portion ###
models = get_models()
results = []

# AQ10 Question set for comparison
question_sets = {
    'All 50': x,
    'Official AQ-10': x_aq10,
}

for set_name, x_set in question_sets.items():
    for name, model in models.items():
        print(f"\n{set_name} - {name}:")
        
        ## Performs cross validation and returns a dictionary of scores for each metric
        scores = cross_validate(model, x_set, y, cv=get_cv(), scoring=['accuracy', 'precision', 'recall', 'f1'], return_train_score=True)

        # Print the means and standard deviations of the scores for each metric
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            test = scores[f'test_{metric}']
            train = scores[f'train_{metric}']
            print(f"{metric}: Test mean = {test.mean():.3f}, Test std = {test.std():.3f}, Train mean = {train.mean():.3f}, Train std = {train.std():.3f}")

        # Store the results
        results.append({
            'Model': name,
            'Questions': set_name,
            'Test Accuracy Mean': scores['test_accuracy'].mean(),
            'Test Accuracy Std': scores['test_accuracy'].std(),
            'Train Accuracy Mean': scores['train_accuracy'].mean(),
            'Train Accuracy Std': scores['train_accuracy'].std(),
            'Test Precision Mean': scores['test_precision'].mean(),
            'Test Precision Std': scores['test_precision'].std(),
            'Train Precision Mean': scores['train_precision'].mean(),
            'Train Precision Std': scores['train_precision'].std(),
            'Test Recall Mean': scores['test_recall'].mean(),
            'Test Recall Std': scores['test_recall'].std(),
            'Train Recall Mean': scores['train_recall'].mean(),
            'Train Recall Std': scores['train_recall'].std(),
            'Test F1 Mean': scores['test_f1'].mean(),
            'Test F1 Std': scores['test_f1'].std(),
            'Train F1 Mean': scores['train_f1'].mean(),
            'Train F1 Std': scores['train_f1'].std()
        })  

results_df = pd.DataFrame(results)
print(results_df.round(3))
results_df.to_csv('../results/aq50_vs_aq10_baseline.csv', index=False)




