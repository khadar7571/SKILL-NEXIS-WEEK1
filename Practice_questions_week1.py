#-----------------PRACTICE QUESTIONS week1------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1) Load a CSV using pandas and print first 10 rows.

df = pd.read_csv('Titanic-Dataset.csv')
print(df.head(10))

# 2) Split dataset into train/test using sklearn

data = df[['Age', 'Fare']].dropna()
X = data[['Age']] 
y = data['Fare']  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3) Train a Linear Regression model and check accuracy.

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")

# 4)Predict ticket fare based on passenger age
user_age = float(input("Enter passenger age: "))

predicted_fare = model.predict([[user_age]])

print(f"Predicted Ticket Fare: ${predicted_fare[0]:.2f}")