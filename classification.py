# -*- coding: utf-8 -*-
"""
Created on Mon May 10 02:19:15 2021

@author: Mert Doğan
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

def printTable(d, name):
    print(f"\n{name} of Attributes")
    for k in d:
        print("\t", k,":", d[k], end="\n")
    print()
        

df = pd.read_csv("Training Data.csv")
print(df, "\n")
print("Dimension: ", df.ndim)
print("Shape: ", df.shape, "\n")

print(df.info(), "\n")
#remove unnecessary features from data
df = df.drop(["Id"],axis=1) 
df = df.drop(["age"],axis=1)
df = df.drop(["married"],axis=1)
df = df.drop(["city"],axis=1)
df = df.drop(["state"],axis=1)
df = df.drop(["profession"],axis=1)

#divide target and data
target=df.risk_flag.values
data=df.drop(["risk_flag"],axis=1)

dtypes={}
sums={}
maxes={}
mins={}
means={}
medians={}

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for c in data.columns:
    if(is_numeric_dtype(data[c])==False):
        data[c] = le.fit_transform(data[c]) #converting non-numbers to numbers
    #getting some information
    sums[c]=str(data[c].sum())
    maxes[c]=str(data[c].max())
    mins[c]=str(data[c].min())
    means[c]=str(data[c].mean())
    medians[c]=str(data[c].median())

print(data, "\n") 
print(data.info(), "\n")     

#find the ratio of target classes to observe balancing
print("Balancing:\n")
print(df['risk_flag'].value_counts())
print("percentage of 0 =", 100*(df['risk_flag'].value_counts()[0] / (df['risk_flag'].value_counts()[0]+df['risk_flag'].value_counts()[1])),"%")
print("percentage of 1 =", 100*(df['risk_flag'].value_counts()[1] / (df['risk_flag'].value_counts()[0]+df['risk_flag'].value_counts()[1])),"%")
  
#print statistics of attributes
printTable(sums, "Sums")
printTable(maxes, "Max Values")
printTable(mins, "Min Values")
printTable(means, "Means")
printTable(medians, "Medians")

#boxplot
data['income'] = data['income'].apply(lambda x: x/1000000)
cols = data.columns.tolist()
cols.remove("house_ownership")
cols.remove("car_ownership")
boxplot = data.boxplot(cols, figsize=(13,8))
cols[0]="income(x1M)"
boxplot.set_xlabel("* income represents 1M times")
data['income'] = data['income'].apply(lambda x: x*1000000)


#creating pipeline
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

pipeline = Pipeline([
    ('normalizer', preprocessing.StandardScaler()), #normalize data
    ('clf', DecisionTreeClassifier(class_weight="balanced")) #classifier
])

#split for train
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state = 26)


#GridSearchCV for specify parameter values and training
from sklearn.model_selection import GridSearchCV

criterion = ['gini', 'entropy']

gscv = GridSearchCV(pipeline, param_grid = {
    'clf__random_state' : [26],
    'clf__criterion' : criterion, 
})

gscv.fit(data_train, target_train)

#Confusion Matrix
from sklearn.metrics import confusion_matrix

target_predict = gscv.predict(data_test)
cm = confusion_matrix(target_test,target_predict)

#Visualization of Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True,linewidths=2,linecolor="red", fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("Confusion Matrix")
plt.show()

#calculate evaluation scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("Accuracy Score: ", accuracy_score(target_test,target_predict))
print("Best Params of GridSearchCV:", gscv.best_params_)
print("Evaluation Score:")
print(classification_report(target_test, target_predict))
