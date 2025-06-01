# ai-medical-prognosis

This repository contains tools and examples for medical prognosis using artificial intelligence and statistical modeling. It covers data feature engineering, risk score calculations, and survival analysis with Cox models and Random Forest classifiers.

**Contents**
**prognosis_feature_engineering.py**: Generates synthetic medical features and visualizes feature interactions.

**risk_scores.py**: Implements calculation of three important clinical risk scores (CHADS-VASC, MELD, ASCVD) with plotting functions.

**survival_analysis_assignment.py**: Performs survival analysis using Cox Proportional Hazards and Random Forest models on mock patient data.

**Script Descriptions**
**1. prognosis_feature_engineering.py**
Creates synthetic medical data (age, systolic blood pressure, cholesterol).

Constructs new interaction features using addition and multiplication.

Visualizes interactions with scatter plots and heatmaps to compare additive vs multiplicative effects.

**2. risk_scores.py**
Functions to calculate key clinical risk scores:

CHADS-VASC: Stroke risk in atrial fibrillation.

MELD: Severity of liver disease.

ASCVD: Cardiovascular disease risk.

Includes example usage with sample input values.

Displays results with a bar chart for easy comparison.

**3. survival_analysis_assignment.py**
Uses mock survival data for demonstration.

Fits a penalized Cox Proportional Hazards model.

Computes Harrell’s C-index to evaluate model concordance.

Builds a Random Forest classifier to predict 1-year survival.

Visualizes feature importance from the Random Forest model.

