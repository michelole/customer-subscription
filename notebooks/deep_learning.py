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

# Based on https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

# %%
# Load the preprocessed data
# Encoding categorical data will be done using keras
import pandas as pd

dataframe = pd.read_excel("data/train_file.xlsx")
print(dataframe.dtypes)
dataframe.head()

# %%
# Ensure tensorflow is installed
import numpy as np
import pandas as pd
import tensorflow as tf

assert tf.__version__ == '2.7.0'

# %%
# Create a target variable
dataframe['target'] = np.where(dataframe['y']=='yes', 1, 0)
dataframe.head()

# %%
# Drop unused columns
dataframe = dataframe.drop(columns=['y'])
dataframe.head()

# %%
# Split into train, validation, and test sets
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
print(len(train), 'training examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# %%
# Create an input pipeline
from notebooks.util import df_to_dataset

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['age'])
print('A batch of targets:', label_batch )

# %%
# Normalize numerical features
from notebooks.util import get_normalization_layer

photo_count_col = train_features['duration']
layer = get_normalization_layer('duration', train_ds)
layer(photo_count_col)
# %%
# Normalize categorical features
from notebooks.util import get_category_encoding_layer

test_type_col = train_features['job']
test_type_layer = get_category_encoding_layer(name='job', dataset=train_ds, dtype='string')
print(test_type_layer(test_type_col))

test_age_col = train_features['campaign']
test_age_layer = get_category_encoding_layer(name='campaign', dataset=train_ds, dtype='int64', max_tokens=5)
print(test_age_layer(test_age_col))
# %%
# Preprocess the data

batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
# %%
# Normalize the data
all_inputs = []
encoded_features = []

# Numerical features
for header in ['age', 'duration']:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

# Integer categorical features
# 'previous'
for header in ['previous', 'campaign']:
    int_categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')

    encoding_layer = get_category_encoding_layer(name='age', dataset=train_ds,
        dtype='int64', max_tokens=5)
    encoded_col = encoding_layer(int_categorical_col)
    all_inputs.append(int_categorical_col)
    encoded_features.append(encoded_col)

# Categorical features
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds,
        dtype='string', max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)
# %%
# # Training the model

# Merge feature inputs
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)

# Creare and compile the model
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

# %%
# Visualize the model
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
# %%
# Train the model
history = model.fit(train_ds, epochs=10, validation_data=val_ds)

# %%
# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

# %%
# Plot learning curve
from matplotlib import pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title("Loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.title("Accuracy")
plt.show()

# %%
# Save the model
model.save('data/customer_predictor')
reloaded_model = tf.keras.models.load_model('data/customer_predictor')
