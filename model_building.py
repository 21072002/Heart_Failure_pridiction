import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
scaler = StandardScaler()

X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Log_reg = LogisticRegression(max_iter = 200)
Log_reg.fit(X_train, y_train)

y_pred = Log_reg.predict(X_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

pickle.dump(Log_reg, open('heart_failure_prediction.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))