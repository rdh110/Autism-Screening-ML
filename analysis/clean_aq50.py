# Clean the AQ50 dataset, put it into a new csv file, and make it ready for analysis. 

# Read the original CSV
# Filder rows that have missing values and remove columns not needed
# Include the 50 question columns + autism diagnosis column
# Save it into a new CSV file as data/cleaned_aq50.csv
# Double check the shape

import pandas as pd

# Read the original CSV
df = pd.read_csv('../data/RAADS-R_Italian_validation_dataset.csv')
print(df.shape)

# Keep only columns with all AQ-50 questions and autism diagnosis
aq50_columns = 