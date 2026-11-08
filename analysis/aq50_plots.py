import pandas as pd
from plots import plot_reduction_curve, plot_question_frequency
from shared_func import AQ10

# Load averages for all models
average_df = pd.read_csv('../results/aq50_reduce_average_all_models.csv',
    dtype={'Question Set Name': str})

# F1 as questions are removed
plot_reduction_curve(average_df, 'F1', 'AQ-10', 
                     'AQ-50: Test F1 as Questions Are Removed',
                     '../plots/aq50_reduction_f1.png')

# Recall as questions are removed
plot_reduction_curve(average_df, 'Recall', 'AQ-10', 
                     'AQ-50: Test Recall as Questions Are Removed',
                     '../plots/aq50_reduction_recall.png')

# Load raw results for all models
all_df = pd.read_csv('../results/aq50_reduce_all_models.csv')
aq10_columns = [f'aq_{i}' for i in AQ10]

# How often each question made the best 10
plot_question_frequency(all_df, aq10_columns, 'AQ-10', 10,
                        'AQ-50: How Often Each Question Was Chosen for the Best 10',
                        '../plots/aq50_question_frequency.png')
