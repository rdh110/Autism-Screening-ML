from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

SEED = 30

# 5-fold cross-validation with 5 repeats for 25 train/test splits
# Stratified keeps same autistic/non-autistic ratio in each fold
# random_state creates a fixed shuffle of data so that results are reproducible
def get_cv():
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)

def get_models():
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        'Random Forest': RandomForestClassifier(random_state=SEED),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(random_state=SEED)),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=SEED),
        'Neural Network': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=30, max_iter=1000, random_state=SEED))
    }
    return models