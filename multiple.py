# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:45:41 2026

@author: Sude
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

df = pd.read_csv('C:/Users/Sude/Desktop/Homeworks/odev_tenis.csv')

df['windy'] = df['windy'].astype(int)
df['play'] = df['play'].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, columns=['outlook'], drop_first=True, dtype=int)

y = df['humidity']
X = df.drop('humidity', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

results_df = pd.DataFrame({'Actual_Humidity': y_test, 'Predicted_Humidity': y_pred})
print("--- Predictions ---")
print(results_df)

print("\n--- Model Evaluation ---")
print(f"R-Squared (R2) Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")

X_opt = sm.add_constant(X)
ols_model = sm.OLS(y, X_opt).fit()

print("\n--- Statistical Summary ---")
print(ols_model.summary())
