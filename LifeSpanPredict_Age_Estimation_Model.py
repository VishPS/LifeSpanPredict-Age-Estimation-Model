import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Loading the data
df = pd.read_csv(r'C:\Users\Vishu\Downloads\Human_Age_Prediction.csv')

# Data Preprocessing
# Handling missing values
df.ffill(inplace=True)

# Splitting 'Blood Pressure' column into 2 columns diastolic and systolic .
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure (s/d)'].str.split('/', expand=True).astype(float)

# Dropping the original 'Blood Pressure' column
df = df.drop(columns=['Blood Pressure (s/d)'])

# Creating new features
df['Blood Pressure Product'] = df['Systolic BP'] * df['Diastolic BP']  

# Age bins (example ranges)
df['Age Group'] = pd.cut(df['Age (years)'], bins=[0, 20, 40, 60, 80, 100], labels=['Young', 'Middle-aged', 'Senior', 'Above-senior', 'Elderly'], right=False)

# Changing value to numerals
df['Physical Activity Level'] = df['Physical Activity Level'].map({
    'High': 1, 
    'Low': 2, 
    'Moderate': 3,
})
df['Smoking Status'] = df['Smoking Status'].map({
    'Former': 1, 
    'Current': 2, 
    'Never': 3,
})
df['Alcohol Consumption'] = df['Alcohol Consumption'].map({
    'Occasional': 1, 
    'Frequent': 2, 
    'None': 3,
})
df['Medication Use'] = df['Medication Use'].map({
    'Occasional': 1, 
    'Regular': 2, 
    'None': 3,
})
df['Income Level'] = df['Income Level'].map({
    'High': 1, 
    'Low': 2, 
    'Medium': 3,
})

# Using LabelEncoder for other columns
le = LabelEncoder()
categorical_columns = ['Gender', 'Education Level', 'Diet', 'Chronic Diseases', 
                       'Family History', 'Cognitive Function', 'Mental Health Status', 
                       'Sleep Patterns', 'Age Group']  

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Age (years)'])  
y = df['Age (years)']  

# Splitting data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# using random forest model
model = RandomForestRegressor(random_state=42)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15, 20, 25, 30, 35],
    'min_samples_split': [2, 3, 5, 10, 15],
    'min_samples_leaf': [1, 2, 3, 4, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2', 0.5]
}

# Using Random Search CV
grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                  cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Calculating results
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying results
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")
