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

# %%
import pandas as pd

data = pd.read_excel("data/train_file.xlsx")
print(data.dtypes)
data.head()

# %%
# Fix object types as category
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category')
data.dtypes

# %%
# Create numerical output
data['y_yes'] = data['y'] == 'yes'
data

# %%
import math

# Group integers
# XXX Assume call duration is in seconds
data['age_group'] = data['age'].apply(lambda x: math.floor(x / 10) * 10)
data['duration_min_group'] = data['duration'].apply(lambda x: math.floor(x / 60 / 10) * 10)
data.head()
# %%

# Find missing values
data.isnull().sum()
# %%
from matplotlib import pyplot as plt

NUMERIC = ['age', 'duration']
CATEGORICAL = ['age_group', 'duration_min_group', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'previous', 'poutcome', 'y']

# Visualize data
for col in NUMERIC:
    data[col].plot.box()
    plt.show()

for col in CATEGORICAL:
    data[col].value_counts().plot.bar()
    plt.title(col)
    plt.show()
# %%
# Explore correlations in numeric data
for col in NUMERIC:
    data.boxplot(column=col, by='y')
    plt.show()

# %%
# Explore correlations in categorical data
for col in CATEGORICAL:
    data.groupby(col)['y_yes'].mean().plot.bar()
    plt.show()
# %%
from itertools import combinations
import seaborn as sns


# Explore correlations among two categorical variables
for (col1, col2) in combinations(CATEGORICAL, 2):
    crosstab = pd.crosstab(data[col1], data[col2], values=data['y_yes'], aggfunc='mean', normalize=True)
    sns.heatmap(crosstab, cmap='OrRd', vmin =0, vmax=0.5)
    plt.show()
