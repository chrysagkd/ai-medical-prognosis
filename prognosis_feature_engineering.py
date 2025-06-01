import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Δημιουργία συνθετικών ιατρικών δεδομένων ---
np.random.seed(42)

# 100 ασθενείς
n_patients = 100

# Παράμετροι: ηλικία (40-80), συστολική πίεση (100-180), χοληστερόλη (150-280)
age = np.random.randint(40, 81, n_patients)
systolic_bp = np.random.randint(100, 181, n_patients)
cholesterol = np.random.randint(150, 281, n_patients)

# Δημιουργία DataFrame
df = pd.DataFrame({
    'Age': age,
    'Systolic_BP': systolic_bp,
    'Cholesterol': cholesterol
})

# --- Συνδυαστικά χαρακτηριστικά ---

# Πρόσθεση (πιο "απλή" αλληλεπίδραση)
df['Age_plus_SBP'] = df['Age'] + df['Systolic_BP']

# Πολλαπλασιασμός (ισχυρότερη αλληλεπίδραση)
df['Age_times_SBP'] = df['Age'] * df['Systolic_BP']

# --- Οπτικοποίηση: σύγκριση πρόσθεσης vs πολλαπλασιασμού ---

plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Age', y='Systolic_BP', size='Age_plus_SBP', data=df, legend=False, sizes=(20, 200))
plt.title('Αλληλεπίδραση με Πρόσθεση\n(Size ~ Age + Systolic_BP)')
plt.xlabel('Ηλικία')
plt.ylabel('Συστολική Πίεση')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Age', y='Systolic_BP', size='Age_times_SBP', data=df, legend=False, sizes=(20, 200), color='red')
plt.title('Αλληλεπίδραση με Πολλαπλασιασμό\n(Size ~ Age x Systolic_BP)')
plt.xlabel('Ηλικία')
plt.ylabel('Συστολική Πίεση')

plt.tight_layout()
plt.show()

# --- Heatmap με πίνακα διασταύρωσης για Age x SBP interaction ---

# Ομαδοποίηση ηλικίας και συστολικής πίεσης σε bins για πιο καθαρό heatmap
df['Age_bin'] = pd.cut(df['Age'], bins=[39, 50, 60, 70, 80], labels=['40-50','51-60','61-70','71-80'])
df['SBP_bin'] = pd.cut(df['Systolic_BP'], bins=[99, 120, 140, 160, 180], labels=['100-120','121-140','141-160','161-180'])

pivot_table = df.pivot_table(index='Age_bin', columns='SBP_bin', values='Age_times_SBP', aggfunc='mean')

plt.figure(figsize=(8,6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='coolwarm')
plt.title('Μέση Τιμή Αλληλεπίδρασης Ηλικίας x Συστολικής Πίεσης')
plt.xlabel('Συστολική Πίεση (mmHg)')
plt.ylabel('Ηλικία (έτη)')
plt.show()
 