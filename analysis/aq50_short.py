import pandas as pd
import numpy as np

aq50_df = pd.read_csv('../data/cleaned_aq50.csv')

# drop label column
x = aq50_df.drop(columns=['autistic'])

# select label column
y = aq50_df['autistic']

# covariance matrix
cov_matrix = np.cov(x, rowvar=False)

eigenvalues = np.linalg.eigvalsh(cov_matrix)[::-1]  # sort in descending order

# Share of total variance explained by each eigenvalue
variance_ratio = eigenvalues / eigenvalues.sum()

cumulative_variance_ratio = np.cumsum(variance_ratio)

print(f"First component explains {variance_ratio[0]:.1%}")
print(f"Components needed for 80%: {np.argmax(cumulative_variance_ratio >= 0.80) + 1}")
print(f"Components needed for 90%: {np.argmax(cumulative_variance_ratio >= 0.90) + 1}")

