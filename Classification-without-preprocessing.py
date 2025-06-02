# -*- coding: utf-8 -*-
"""
Mushroom Classification with 5-Iteration Averaging
"""

import os
os.chdir("C:\\Users\\behan\\Desktop\\DataMining project")

# Library Importing
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

# Data importing
mushrooms = pd.read_csv('primary_data.csv', sep=';')
print("\nData Summary:")
mushrooms.info()
print("\nMissing Values (%):")
print(mushrooms.isnull().mean()*100)

# Separate features and target
X = mushrooms.drop('class', axis=1) 
y = mushrooms['class']  # Target (poisonous 'p' or edible 'e')

# Categorical transformation 
encoder = ce.OrdinalEncoder(
    cols=['family', 'name', 'cap-shape', 'Cap-surface', 'cap-color', 'does-bruise-or-bleed', 
          'gill-attachment', 'gill-spacing', 'gill-color', 'stem-height', 'stem-width', 
          'stem-root', 'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring', 
          'ring-type', 'Spore-print-color', 'habitat', 'season'],
    drop_invariant=True
)

# Encode categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = ce.OrdinalEncoder(cols=categorical_cols, drop_invariant=True)
X_encoded = encoder.fit_transform(X)

# Initialize storage for metrics
rf_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'confusion': []}
knn_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'confusion': []}

# Run 5 iterations
n_iterations = 5

for i in range(n_iterations):

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=i
    )
    
    # Standardization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Random Forest ---
    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    # Store metrics
    rf_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    rf_metrics['precision'].append(precision_score(y_test, y_pred, pos_label='p'))
    rf_metrics['recall'].append(recall_score(y_test, y_pred, pos_label='p'))
    rf_metrics['f1'].append(f1_score(y_test, y_pred, pos_label='p'))
    rf_metrics['confusion'].append(confusion_matrix(y_test, y_pred))
    
    
    
    # --- KNN ---
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    
    # Store metrics
    knn_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    knn_metrics['precision'].append(precision_score(y_test, y_pred, pos_label='p'))
    knn_metrics['recall'].append(recall_score(y_test, y_pred, pos_label='p'))
    knn_metrics['f1'].append(f1_score(y_test, y_pred, pos_label='p'))
    knn_metrics['confusion'].append(confusion_matrix(y_test, y_pred))
    


# Calculate averages
def get_averages(metrics):
    return {
        'accuracy': np.mean(metrics['accuracy']),
        'precision': np.mean(metrics['precision']),
        'recall': np.mean(metrics['recall']),
        'f1': np.mean(metrics['f1']),
        'confusion': np.mean(metrics['confusion'], axis=0).round().astype(int)
    }

rf_avg = get_averages(rf_metrics)
knn_avg = get_averages(knn_metrics)

# Print final averaged results
print("\n\n===THE AVEREGED RESULT OF 5 ITERATIONS ===")

print("\nRandom Forest (Averaged over 5 runs):")
print(f"Accuracy:    {rf_avg['accuracy']:.4f}")
print(f"Precision:   {rf_avg['precision']:.4f}")
print(f"Recall:      {rf_avg['recall']:.4f}")
print(f"F1-Score:    {rf_avg['f1']:.4f}")
print("\nAverage Confusion Matrix:")
print(rf_avg['confusion'])

print("\nKNN (Averaged over 5 runs):")
print(f"Accuracy:    {knn_avg['accuracy']:.4f}")
print(f"Precision:   {knn_avg['precision']:.4f}")
print(f"Recall:      {knn_avg['recall']:.4f}")
print(f"F1-Score:    {knn_avg['f1']:.4f}")
print("\nAverage Confusion Matrix:")
print(knn_avg['confusion'])

print("\nExecution completed!")