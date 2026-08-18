# ------------------Assignments_question_week1------------------


# 1) Load a dataset using Pandas and summarize basic stats (.info(), .describe()).


import pandas as pd
df = pd.read_csv('Titanic-Dataset.csv')
print("--- Dataframe Info ---")
df.info()
print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())
print("\n--- Summary Statistics (All Columns) ---")
print(df.describe(include='all'))


#2) Handle missing data using mean/median imputation


print("\n--- Missing Data Before Imputation ---")
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print("\n--- Missing Data After Imputation ---")
print(df.isnull().sum())


#3)Encode categorical variables using LabelEncoder & OneHotEncoder.


print("\n--- Encoding Categorical Variables ---")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
le = LabelEncoder()
df['Sex_Encoded'] = le.fit_transform(df['Sex'])  # male -> 1, female -> 0
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop first to prevent multicollinearity
embarked_encoded = ohe.fit_transform(df[['Embarked']])
embarked_df = pd.DataFrame(
    embarked_encoded, 
    columns=ohe.get_feature_names_out(['Embarked']))
df_final = pd.concat([df, embarked_df], axis=1)
print(df_final[['Sex', 'Sex_Encoded', 'Embarked'] + list(embarked_df.columns)].head())
 
