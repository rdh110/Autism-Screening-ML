import numpy as np
import matplotlib.pyplot as plt

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