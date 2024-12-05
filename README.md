# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries .
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5.Determine training and test data set.
6.Apply decision tree Classifier on to the dataframe.
7.Get the values of accuracy and data prediction

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:prabanjan.m 
RegisterNumber:24900428
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree  # Import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
*/
```

## Output:
![Screenshot 2024-12-05 191842](https://github.com/user-attachments/assets/1f2f0467-7d4e-4e8f-ace1-7d562762f1e0)
![Screenshot 2024-12-05 191833](https://github.com/user-attachments/assets/bed32305-72f6-4a5d-b44c-bb7b9c67fb6d)
![Screenshot 2024-12-05 191820](https://github.com/user-attachments/assets/41ff4290-071b-42e9-98ea-2536485ae861)
![Screenshot 2024-12-05 191812](https://github.com/user-attachments/assets/388104a0-6753-45bb-b3ff-6f3c10400ee9)
![Screenshot 2024-12-05 191759](https://github.com/user-attachments/assets/0f7fa445-2309-4d2b-a4b4-922f7969dabf)
![Screenshot 2024-12-05 191751](https://github.com/user-attachments/assets/60d95b1d-aca3-4766-a399-1aafe3c70867)
![Screenshot 2024-12-05 191851](https://github.com/user-attachments/assets/81b6718d-73d3-44b3-a5ec-f138dacbcdc8)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
