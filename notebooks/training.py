# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [md]
# # Predicting costumer subscription
# ## Training ML Models

# %%
# Load the preprocessed data
import pandas as pd

data = pd.read_pickle("data/preprocessed_data.pkl")
data.head()

# %%
# Since our dataset is imbalanced, calculate a majority baseline.
from sklearn.dummy import DummyClassifier

SEED = 42

majority = DummyClassifier(random_state=SEED)

# %%
# Use SVM to train the model.
# SVM typically leads to near-optimal results in linearly separable problems.
from sklearn.svm import LinearSVC

svm = LinearSVC(dual=False, random_state=SEED)

# %%
# Use kNN to train the model.
# kNN may work well when a general function cannot be learned
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# %%
# Use a random forest to train the model.
# Random forests may offer explainable solutions.
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=SEED)

# %%
# Use logistic regression to train the model.
# Logistic regression is a strong baseline for binary classification problems
# and also provides an explainable model.
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=SEED, max_iter=1000)

# %%
# Use a boosting algorithm to train the model.
# Boosting may generalize better on test data.
from sklearn.ensemble import GradientBoostingClassifier

#  HistGradientBoostingClassifier(categorical_features=[0])
gb = GradientBoostingClassifier(random_state=SEED)

# %%
# Drop columns used only for visualization
data = data.drop(['y_yes', 'age_group', 'duration_min_group'], axis=1)

# %%
# Encode categorical data

dummies = pd.get_dummies(data)
dummies.head()

# %%
# Create training and test sets
X_train = dummies.drop(['y_no', 'y_yes'], axis=1)
y_train = dummies['y_yes']

# %%
# Persist data for reuse
X_train.to_pickle("data/X_train.pkl")
y_train.to_pickle("data/y_train.pkl")

# %%
# Iterate over classifier to generate repors
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

from notebooks.util import plot_confusion_matrix

SCORING = 'f1_macro'

classifiers = {
    'Majority': majority,
    'SVM': svm,
    'kNN': knn,
    'Random Forest': rf,
    'LR': lr,
    'Gradient Boosting': gb
    }

results = pd.DataFrame(columns=[SCORING, "model"])

for k, v in classifiers.items():    
    # Cross-validate the dataset.
    clf_scores = cross_val_score(v, X_train, y_train, scoring=SCORING, cv=10)
    results = pd.concat([results, pd.DataFrame({SCORING: clf_scores, 'model': k})])
    print(f"{k}: {clf_scores.mean()} +- {clf_scores.std()}")

    # Generate confusion matrix.
    pred = cross_val_predict(v, X_train, y_train, cv=10)
    plot_confusion_matrix(y_train, pred, k)

    # Display classification report.
    print(classification_report(y_train, pred))

# %%
# Compare overall results using boxplot
from matplotlib import pyplot as plt

results.boxplot(column=[SCORING], by='model',# positions=[1, 3, 2],
    showmeans=True, figsize=(7, 5))
plt.ylim([0.4, 1.0])

# %%
# Random Forest: plot one decision tree to explain the model
from sklearn.tree import plot_tree

rf = RandomForestClassifier(n_estimators=10, random_state=SEED)
print(cross_val_score(rf, X_train, y_train, scoring=SCORING, cv=10).mean())
rf.fit(X_train, y_train)

ESTIMATOR = 0

fig = plt.figure(figsize=(150, 100))
plot_tree(rf.estimators_[ESTIMATOR], max_depth=3, feature_names=X_train.columns,
    class_names=["no", "yes"], filled=True, proportion=True, rounded=True)

# %%
# Feature importance
import numpy as np

classifiers = {
    'Random Forest': rf,
    'Gradient Boosting': gb
    }

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': clf.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    top = importance.head(10)
    top.plot.bar()
    plt.xticks(np.arange(len(top)), labels=top['feature'])
    plt.title(f"Top {len(top)} features for {name}")
    plt.show()

# %%
# # Predict on test data
test = pd.read_excel("data/test_file.xlsx")
print(test.dtypes)
test.head()

# %%
# Encode test data the same as train data
test_dummies = pd.get_dummies(test).reindex(columns=X_train.columns, fill_value=0)
test_dummies.head()

# %%
# Predict and save results
test['y_pred'] = gb.predict(test_dummies)
test.to_excel("data/test_file_pred.xlsx")
