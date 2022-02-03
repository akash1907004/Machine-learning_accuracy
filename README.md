# Machine-learning_accuracy
# This code is based on the dataset.We load the dataset and we found the best model to incorporate in the app

import pandas as pd
# load the dataset
data = pd.read_csv('Desktop/sri ramakrishna engineering college blood bank.csv')
data.head(5)

# No of rows and columns
data.shape
# Information of the data
data.describe()
# rename the bloodgroup(because it was the target)
data.rename(
    columns={'Blood group': 'target'},
    inplace=True
)
# Print out the first 2 rows
data.head(2)
#we have to see the counts in target
data.target.value_counts(normalize=True).round(3)
from sklearn.model_selection import train_test_split

# Split transfusion DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns='target'),
    data.target,
    test_size=0.25,
    random_state=42
)

# Print out the first 2 rows of X_train
X_train.head(2)
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1): 
# Print idx and transform
    print(f'{idx}. {transform}')
# Check the values
 X_train.var().round(3)
 import numpy as np

# Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = 'Frequency'

# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['frequency_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns=col_to_normalize, inplace=True)

# Check the variance for X_train_normed
X_train_normed.var().round(3)
# It will show the accuracy of the models.
from sklearn import linear_model

# Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

# Train the model
logreg.fit(X_train_normed, y_train)

# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')
