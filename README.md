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
## Initial data set:
![exp 6 ml 1](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/2fb1cd6d-4da0-4245-8087-0f3dc6adbadf)
## Data info:
![exp 6 ml 2](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/b06006a8-34b7-4a52-ad76-bd925c9ce6c3)
## Optimization of null values:
![exp 6 ml 3](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/f585db85-5818-4437-bcc3-2740f900914b)
## Assignment of x and y values:
![exp 6 ml 4](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/f3eddb41-9232-4111-8e3f-648033a2082e)
![exp 6 ml 5](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/5afe85e6-d8ca-42bc-8e09-029d80ec2947)
## Converting string literals to numerical values using label encoder:
![exp 6 ml 6](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/46f2a95f-c579-4e6b-a936-5eb265a330fa)
## Accuracy:
![exp 6 ml 7](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/4a1170b7-70a6-4382-b715-c0b8ef3f5640)
## Prediction:
![exp 6 ml 8](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541549/404d7a22-42f9-45c2-a40c-34df17264900)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
