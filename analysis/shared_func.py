from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

SEED = 30

# Official AQ-10 Questions
AQ10 = [5, 20, 27, 28, 31, 32, 36, 37, 41, 45]

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

# outer split for rotating test sets so that every person is in the test set once
# used to score quetion sets on people the question selection hasnt seen
def get_outer_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# inner 5-fold cross-validation for testing scores
# used to choose which questions to keep or remove
def get_selection_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# train on selected questions and then score on the test
def test_scores(model, x_train, y_train, x_test, y_test, questions):
    model.fit(x_train[questions], y_train)
    predictions = model.predict(x_test[questions])
    return {
        'Accuracy': accuracy_score(y_test, predictions),
        'F1': f1_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions)
    }

def remove_questions(model, x, y, cv):
    # start with all questions
    remaining_questions = list(x.columns)
    history = []

    # keep going until 1 question remaining
    while len(remaining_questions) > 1:
        best_score = -1
        least_important_question = None

        # try removing each question and see the effect on cross validation score
        for question in remaining_questions:
            trial = [q for q in remaining_questions if q != question]
            score = cross_val_score(model, x[trial], y, cv=cv, scoring='f1')
            mean_score = score.mean()
            if mean_score > best_score:
                best_score = mean_score
                least_important_question = question

        remaining_questions.remove(least_important_question)
        history.append({
            'Questions Left': len(remaining_questions),
            'Removed Question': least_important_question,
            'Best Score': best_score,
            'Question Set': list(remaining_questions)
        })

    return remaining_questions, history

# rotating nested version of question removal for feature selection
# each time with different group of people
def nested_remove_questions(model, x, y, outer_cv, inner_cv, baseline_name, baseline_columns):
    rows = []
    fold = 1

    # iterate over each fold of the outer cross-validation
    for train_index, test_index in outer_cv.split(x, y):
        print(f'Fold {fold}')

        # split into this fold's training and test sets
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # choose questions using only the training set
        final_question, history = remove_questions(model, x_train, y_train, inner_cv)

        # question sets to score
        question_sets = {
            'All': list(x.columns),
             baseline_name : baseline_columns
        }

        # for every reduced set
        for step in history:
            question_sets[step['Questions Left']] = step['Question Set']

        # score each question set on fold's test set
        for set_name, questions in question_sets.items():
            scores = test_scores(model, x_train, y_train, x_test, y_test, questions)

            rows.append({
                'Fold': fold,
                'Question Set Name': set_name,
                'Number of Questions': len(questions),
                'Accuracy': scores['Accuracy'],
                'F1': scores['F1'],
                'Precision': scores['Precision'],
                'Recall': scores['Recall'],
                'Questions': questions,
            })
        fold += 1

    return rows
