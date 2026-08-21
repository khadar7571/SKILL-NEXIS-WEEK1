
#-------------------ASSIGNMENTS QUESTIONS WEEK 2--------------------------

import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1) Build Linear Regression model on housing dataset (predict price)

df = pd.read_csv('Housing.csv')
X = df.drop('price', axis=1)
y = df['price']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Housing Dataset")
print("MSE:", mse)
print("R2 Score:", r2)
print("First 5 Predicted Prices:", y_pred[:5])


# 2) Train Logistic Regression on Titanic dataset for survival prediction


df = pd.read_csv('Titanic-Dataset.csv')
df.drop(['Cabin'], axis=1, inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']
X = pd.get_dummies(X,columns=['Sex', 'Embarked'],drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\nTitanic Dataset")
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)