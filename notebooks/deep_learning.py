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
# ## Training a Deep Learning Model

# %%
# Load the preprocessed data
# Encoding categorical data will be done using keras
import pandas as pd

data = pd.read_pickle("data/preprocessed_data.pkl")
data.head()
# %%
