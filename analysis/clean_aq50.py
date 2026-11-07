import pandas as pd

# Read the original CSV
df = pd.read_csv('../data/RAADS-R_Italian_validation_dataset.csv')

# People who completed AQ-50
completed_aq50 = df['aq_total'].notnull()
aq_df = df[completed_aq50]

# Build list of 50 question columns
question_columns = []
for i in range(1, 51):
    question_columns.append(f'aq_{i}')

# Keep the 50 questions plus diagnosis label
aq50_df = aq_df[question_columns + ['autistic']]

# Change answers to int
aq50_df = aq50_df.astype(int)

# Create csv of cleaned data
aq50_df.to_csv('../data/cleaned_aq50.csv', index=False)