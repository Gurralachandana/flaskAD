#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


df =pd.read_csv("Tele-communication Churn.csv")
df


# In[3]:


df.info()


# In[4]:


df.head()


# ### Observations from the Dataset:
# - 20 columns with a mix of categorical (state, area.code, voice.plan, intl.plan, churn) and numerical (account.length, intl.mins, day.calls, etc.).
# - No missing values in any column.
# - Redundant column: Unnamed: 0 (index-like, can be dropped).
# - Target variable: churn (binary classification with "yes"/"no" values).
# - Categorical features: state, area.code, voice.plan, intl.plan, churn.
# - Potential feature engineering: Converting categorical variables to numerical, scaling, and handling class imbalance.

# In[5]:


# Drop the redundant column 'Unnamed: 0'
df_cleaned = df.drop(columns=['Unnamed: 0'])

# Convert categorical variables to numerical format
df_cleaned['voice.plan'] = df_cleaned['voice.plan'].map({'yes': 1, 'no': 0})
df_cleaned['intl.plan'] = df_cleaned['intl.plan'].map({'yes': 1, 'no': 0})
df_cleaned['churn'] = df_cleaned['churn'].map({'yes': 1, 'no': 0})

# One-hot encoding for 'state' and 'area.code'
df_cleaned = pd.get_dummies(df_cleaned, columns=['state', 'area.code'], drop_first=True)

# Display cleaned dataset information
df_cleaned.info(), df_cleaned.head()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df_cleaned['churn'], palette="pastel")
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Correlation heatmap of top features
plt.figure(figsize=(12, 6))
corr_matrix = df_cleaned.corr()
sns.heatmap(corr_matrix[['churn']].sort_values(by='churn', ascending=False), annot=True, cmap="coolwarm")
plt.title("Feature Correlation with Churn")
plt.show()


# # STANDARDIZATION

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features and target variable
X = df_cleaned.drop(columns=['churn'])
y = df_cleaned['churn']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Output shape after preprocessing
X_train_scaled.shape, X_test_scaled.shape


# In[8]:


df_cleaned


# In[10]:


get_ipython().system('pip install xgboost')


# # MODEL TRAINING AND EVALUATION

# 

# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Initialize models with class weighting to handle imbalance
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb_clf = XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)

# Train models
log_reg.fit(X_train_scaled, y_train)
rf_clf.fit(X_train_scaled, y_train)
xgb_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_rf = rf_clf.predict(X_test_scaled)
y_pred_xgb = xgb_clf.predict(X_test_scaled)

# Evaluation metrics
log_report = classification_report(y_test, y_pred_log)
rf_report = classification_report(y_test, y_pred_rf)
xgb_report = classification_report(y_test, y_pred_xgb)

log_auc = roc_auc_score(y_test, y_pred_log)
rf_auc = roc_auc_score(y_test, y_pred_rf)
xgb_auc = roc_auc_score(y_test, y_pred_xgb)

log_report, log_auc, rf_report, rf_auc, xgb_report, xgb_auc


# In[12]:


# Train only Logistic Regression and Random Forest (excluding XGBoost)
log_reg.fit(X_train_scaled, y_train)
rf_clf.fit(X_train_scaled, y_train)
xgb_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_rf = rf_clf.predict(X_test_scaled)
y_pred_xgb = xgb_clf.predict(X_test_scaled)

# Evaluation metrics
log_report = classification_report(y_test, y_pred_log)
rf_report = classification_report(y_test, y_pred_rf)
xgb_report = classification_report(y_test, y_pred_xgb)


log_auc = roc_auc_score(y_test, y_pred_log)
rf_auc = roc_auc_score(y_test, y_pred_rf)
xgb_auc = roc_auc_score(y_test, y_pred_xgb)

log_report, log_auc, rf_report, rf_auc, xgb_report, xgb_auc


# # MODEL TRAINING AND EVALUATION

# In[12]:


# Re-initialize models
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb_clf = XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)



# Train models
log_reg.fit(X_train_scaled, y_train)
rf_clf.fit(X_train_scaled, y_train)
xgb_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_rf = rf_clf.predict(X_test_scaled)
y_pred_xgb = xgb_clf.predict(X_test_scaled)

# Evaluation metrics
log_report = classification_report(y_test, y_pred_log)
rf_report = classification_report(y_test, y_pred_rf)
xgb_report = classification_report(y_test, y_pred_xgb)


log_auc = roc_auc_score(y_test, y_pred_log)
rf_auc = roc_auc_score(y_test, y_pred_rf)
xgb_auc = roc_auc_score(y_test, y_pred_xgb)

log_report, log_auc, rf_report, rf_auc, xgb_report, xgb_auc


# In[13]:


# Re-import necessary metrics
from sklearn.metrics import classification_report, roc_auc_score

# Evaluate models
log_report = classification_report(y_test, y_pred_log)
rf_report = classification_report(y_test, y_pred_rf)
xgb_report = classification_report(y_test, y_pred_xgb)


log_auc = roc_auc_score(y_test, y_pred_log)
rf_auc = roc_auc_score(y_test, y_pred_rf)
xgb_auc = roc_auc_score(y_test, y_pred_xgb)

log_report, log_auc, rf_report, rf_auc, xgb_report, xgb_auc


# # K fold cross validation

# In[14]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

# Initialize Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
    print(f"Mean AUC for {model.__class__.__name__}: {scores.mean():.4f}")

# Evaluate models
evaluate_model(log_reg, X_train_scaled, y_train)
evaluate_model(rf_clf, X_train_scaled, y_train)
evaluate_model(xgb_clf, X_train_scaled, y_train)


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define parameter grids for each model
param_grid_log = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0]
}

# Perform Grid Search for each model
def perform_grid_search(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best AUC: {grid_search.best_score_}")
    return grid_search.best_estimator_

# Perform tuning
best_log_reg = perform_grid_search(log_reg, param_grid_log)
best_rf_clf = perform_grid_search(rf_clf, param_grid_rf)
best_xgb_clf = perform_grid_search(xgb_clf, param_grid_xgb)


# In[15]:


import joblib

# Assuming best_rf_clf is your chosen model from the hyperparameter tuning
model_filename = 'random_forest_model.pkl'

# Save the model
joblib.dump(best_rf_clf, model_filename)
print(f'Model saved to {model_filename}')

# To load the model later for inference
# loaded_model = joblib.load('random_forest_model.pkl')


# In[ ]:




