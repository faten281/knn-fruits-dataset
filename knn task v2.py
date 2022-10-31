# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:18:55 2022

@author: abd_i
"""
#import Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



#get DataSet
fruits = pd.read_csv("fruit_data_with_colours.csv")
fruits.head()
Fruit_name =  dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique() ))
print(type(Fruit_name))
print(Fruit_name)


#get independent_variable (Features)
X = fruits[['mass','width','height']]
X.head(10)

#get dependent_variable (Response)
y= fruits['fruit_label']
y.head(10)


#Split data 
X_train,X_test,y_train,y_test =  train_test_split(X,y,random_state= 42)


#classifier and accuracy
accs_lst= []
for k in range(1, 15):
    classifier = KNeighborsClassifier(n_neighbors=k)
    #train data
    classifier.fit(X_train,y_train)

    #test data
    classifier.score(X_test,y_test)

    #Get predicted response 
    y_predict = classifier.predict(X_test)

    #accuracy
    acc = accuracy_score(y_test, y_predict)

    #add to list
    accs_lst.append(acc)
    print(f"Accuracy at k= {k} is {acc}")



print(f"Max accuracy is {max(accs_lst)} at k = {accs_lst.index(max(accs_lst)) + 1}")













