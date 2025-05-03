import os
from nbformat import v4 as nbf

# Ensure folder exists
os.makedirs("MGH_DATASET", exist_ok=True)

# Create a new Jupyter notebook object
notebook = nbf.new_notebook()
cells = []

# Add notebook title
cells.append(nbf.new_markdown_cell("# Heart Disease Analysis and Prediction Model"))

# 1. Imports and setup
cells.append(nbf.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
"""))

# 2. Load dataset
cells.append(nbf.new_code_cell("""\
file_path = "mgh dataset.csv"
df = pd.read_csv(file_path)
df = df.dropna()
"""))

# 3. Explore data
cells.append(nbf.new_code_cell("""\
print("Shape:", df.shape)
print(df.info())
print(df.describe())
print("Missing values:\\n", df.isnull().sum())
"""))

# 4. Histograms
cells.append(nbf.new_code_cell("""\
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()
"""))

# 5. Correlation heatmap
cells.append(nbf.new_code_cell("""\
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
"""))

# 6. Target countplot
cells.append(nbf.new_code_cell("""\
sns.countplot(data=df, x='TenYearCHD', palette='Set2')
plt.title("Distribution of 10-Year CHD")
plt.tight_layout()
plt.show()
"""))

# 7. CHD by sex
cells.append(nbf.new_code_cell("""\
sns.countplot(data=df, x='sex', hue='TenYearCHD', palette='Set1')
plt.title("Heart Disease by Sex")
plt.tight_layout()
plt.show()
"""))

# 8. Boxplots
cells.append(nbf.new_code_cell("""\
num_features = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
for feature in num_features:
    sns.boxplot(data=df, x='TenYearCHD', y=feature, palette='Set3')
    plt.title(f"{feature} vs TenYearCHD")
    plt.tight_layout()
    plt.show()
"""))

# 9. KDE Plot
cells.append(nbf.new_code_cell("""\
sns.kdeplot(data=df[df['TenYearCHD'] == 0]['totChol'], label='No CHD', fill=True)
sns.kdeplot(data=df[df['TenYearCHD'] == 1]['totChol'], label='CHD', fill=True)
plt.title("Cholesterol Distribution by CHD Status")
plt.xlabel("Total Cholesterol")
plt.legend()
plt.tight_layout()
plt.show()
"""))

# 10. Preprocessing
cells.append(nbf.new_code_cell("""\
features = [
    'age', 'sex', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
]
target = 'TenYearCHD'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
"""))

# 11. Train model
cells.append(nbf.new_code_cell("""\
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))
"""))

# 12. ROC Curve
cells.append(nbf.new_code_cell("""\
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)
print("ROC AUC Score:", auc_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""))

# 13. Save model
cells.append(nbf.new_code_cell("""\
joblib.dump(model, 'heart_disease_prediction_model.pkl')
print("Model saved as heart_disease_prediction_model.pkl")
"""))

# Build notebook
notebook['cells'] = cells

# Save notebook file
output_path = "MGH_DATASET/Heart_Disease_Analysis.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(nbf.writes(notebook))

print(f"Notebook saved to: {output_path}")
