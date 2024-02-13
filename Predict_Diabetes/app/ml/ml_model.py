import pandas as pd
import numpy as np  #import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the data from the CSV file
df = pd.read_csv('C:/Users/HP/Desktop/Code__project__master/OnlineInternship/Project2__Diabetes/Predict_Diabetes/app/ml/diabetes.csv')

# Split the data into features and target variable
data = df.iloc[: , :-1]
labels = df.iloc[: , -1]
#corriger les outliers
outlier_col = ['Glucose', 'BloodPressure','SkinThickness' , 'Insulin', 'BMI']
for i in outlier_col:
    df[i] = df[i].replace(0 , np.nan)
for i in outlier_col:
    df[i].fillna(df[i].median() , inplace=True)

# Split the data into training and test sets
data_train , data_test , labels_train , labels_test = train_test_split(data , labels , test_size=0.2 , random_state=0)

# Create an instance of Logistic Regression model
lrmodel = LogisticRegression(random_state=0 , max_iter=700)

# Train the model
lrmodel.fit(data_train , labels_train)

# Pickle the model
pickle.dump(lrmodel, open('ml_model.sav', 'wb'))
pickle.dump(data.columns, open('columns.sav', 'wb'))