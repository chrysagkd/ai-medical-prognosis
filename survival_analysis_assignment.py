import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier

# Mock dataset
data = pd.DataFrame({
    'time': [100, 200, 300, 400, 500],
    'status': [1, 0, 1, 0, 1],
    'age': [50, 60, 45, 70, 55],
    'bili': [1.2, 2.3, 0.8, 3.1, 1.5],
    'chol': [180, 190, 170, 210, 200],
    'albumin': [3.5, 3.0, 3.8, 2.9, 3.4],
    'copper': [10, 12, 9, 15, 11],
    'alk.phos': [100, 110, 90, 130, 105],
    'ast': [40, 45, 35, 50, 42],
    'trig': [150, 160, 140, 170, 155],
    'platelet': [200, 210, 190, 220, 205],
    'protime': [12, 14, 11, 15, 13],
    'edema': [0, 1, 0, 1, 0],
    'stage': [1, 2, 1, 3, 2],
    'trt': [0, 1, 0, 1, 0]
})

# Αφαίρεση της στήλης trt
data = data.drop(columns=['trt'])

# Κανονικοποίηση συνεχών χαρακτηριστικών
continuous_cols = ['age', 'bili', 'chol', 'albumin', 'copper',
                   'alk.phos', 'ast', 'trig', 'platelet', 'protime']

mean_vals = data[continuous_cols].mean()
std_vals = data[continuous_cols].std()

data[continuous_cols] = (data[continuous_cols] - mean_vals) / std_vals

# One-Hot Encoding κατηγορικών
data_enc = pd.get_dummies(data, columns=['edema', 'stage'], drop_first=False)

# Fit CoxPH Model με penalizer
cph = CoxPHFitter(penalizer=0.1)
cph.fit(data_enc, duration_col='time', event_col='status')
cph.print_summary()

# Υπολογισμός Harrell's C-index
pred = cph.predict_partial_hazard(data_enc)
c_index = concordance_index(data_enc['time'], -pred, data_enc['status'])
print("C-index:", c_index)

# Random Forest για 1-year Survival
def label_one_year_survival(df):
    return ((df['time'] > 365) | (df['status'] == 0)).astype(int)

y = label_one_year_survival(data_enc)
X = data_enc.drop(columns=['time', 'status'])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

acc = rf.score(X, y)
print("Accuracy (1-year survival):", acc)

# Variable Importance Plot
importances = rf.feature_importances_
features = X.columns

vimp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
vimp.plot(kind='bar')
plt.title("Variable Importance (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
