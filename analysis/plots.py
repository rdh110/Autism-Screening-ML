import numpy as np
import matplotlib.pyplot as plt
import ast
from matplotlib.patches import Patch
import pandas as pd

# Create pca plot
def plot_pca(variance_ratio, title, save_path):
    # Create bar plot of variance explained by each component
    components = np.arange(1, len(variance_ratio) + 1) # 1, 2, ..., n_components
    cumulative = np.cumsum(variance_ratio)

    plt.figure(figsize=(9, 4.5))

    # Bars
    plt.bar(components, variance_ratio * 100, label ='Each component')

    # Line
    plt.plot(components, cumulative *100, color= 'red', marker='o', markersize = 5, label = 'Cumulative')

    # Reference lines for 80% and 90%
    for level in (80, 90):
        plt.axhline(y=level, color='gray', linestyle='--', linewidth=1)
        plt.text(0.5, level + 1, f'{level}%', color='gray')

    plt.title(title)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.ylim(0, 105)
    plt.legend(loc='center right')
    plt.yticks(np.arange(0, 110, 10))
    plt.tight_layout()


    plt.savefig(save_path, dpi=300)
    plt.close()

MODEL_STYLES = {
    'Logistic Regression': {'color': 'tab:blue', 'marker': 'o'},
    'Support Vector Machine': {'color': 'tab:orange', 'marker': 's'},
    'Decision Tree': {'color': 'tab:purple', 'marker': '^'},
    'Random Forest': {'color': 'tab:green', 'marker': 'D'},
    'Neural Network': {'color': 'tab:red', 'marker': 'v'}
}

def plot_reduction_curve(average_df, metric, baseline_name, title, save_path):
    plt.figure(figsize=(9, 4.5))

    for model, style in MODEL_STYLES.items():
        rows = average_df[average_df['Model'] == model]

        # reduced sets are named by their number of quesitons
        reduced = rows[rows['Question Set Name'].str.isnumeric()]

        n_questions = reduced['Question Set Name'].astype(int)
        scores = reduced[f'{metric} mean']

        # add the full question set as starting point
        full = rows[rows['Question Set Name'] == 'All']
        x = list(n_questions) + [n_questions.max() + 1]
        y = list(scores) + list(full[f'{metric} mean']) 

        # sort poits by number of questions
        order = np.argsort(x)
        x = np.array(x)[order]
        y = np.array(y)[order]

        # draw this model's line
        plt.plot(x, y, color=style['color'], linewidth=2, label=model)

        # official short form
        base = rows[rows['Question Set Name'] == baseline_name]
        base_n = 10 if baseline_name == 'AQ-10' else 14
        plt.plot(base_n, base[f'{metric} mean'], marker=style['marker'], markersize=9,
                 markerfacecolor='white', markeredgecolor=style['color'],
                 markeredgewidth=2, linestyle='None')
    
    # one legend exlpaining hollow markers
    plt.plot([], [], marker='o', markersize=9, markerfacecolor='white',
              markeredgecolor='grey', markeredgewidth=2, linestyle='None',
                label=f'Official {baseline_name} (same color = same model)')

    plt.axvline(base_n, color='lightgray', linestyle='--', linewidth=1, zorder=0)

    # reverse the x-axis so it reads left to right as questions are removed
    plt.xlim(x.max() + 1, 0.5)
    plt.xticks([50, 40, 30, 10, 5, 1])

    plt.title(title)
    plt.xlabel('Number of Questions')
    plt.ylabel(f'Test {metric} (mean of 5 folds)')
    plt.legend(loc = 'lower left', frameon=False, fontsize = 8)
    plt.grid(axis='y', color='lightgray',linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_question_frequency(all_df, baseline_columns, baseline_name, n_kept, title, save_path):
    # the best-n sets chosen in every run (every model and fold), not official set
    runs = all_df[(all_df['Number of Questions']== n_kept) &
                  (all_df['Question Set Name'] != baseline_name)]
    n_runs = len(runs)

    # count how many runs each question appeared in
    counts = {}
    for question_list in runs['Questions']:
        for question in ast.literal_eval(question_list):
            counts[question] = counts.get(question, 0) + 1

    # include every question, including ones never chosen, sorted most to least
    all_questions = ast.literal_eval(all_df[all_df['Question Set Name'] == 'All']['Questions'].iloc[0])
    frequency = pd.Series({q: counts.get(q, 0) for q in all_questions}).sort_values(ascending=False)

    # officla short-form questions in orange, all others in blue
    colors = ['tab:orange' if q in baseline_columns else 'tab:blue' for q in frequency.index]
    labels = [q.split('_')[1] for q in frequency.index]

    plt.figure(figsize=(12, 4.5))
    plt.bar(range(len(frequency)), frequency.values, color=colors)
    plt.xticks(range(len(frequency)), labels, fontsize=7)

    legend_items = [Patch(color = 'tab:orange', label=f'In official {baseline_name}'),
                    Patch(color = 'tab:blue', label=f'Not in {baseline_name}')]

    plt.title(title)
    plt.xlabel('Question Number')
    plt.ylabel(f'Times in best {n_kept} (out of {n_runs} runs)')
    plt.grid(axis='y', color='lightgray', linewidth=0.5)
    plt.legend(handles=legend_items, frameon=False)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    