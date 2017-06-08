# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:14:11 2017

@author: Douglas
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import cross_validation
from sklearn import tree

movies = pd.read_csv('data/movie_metadata.csv')

movies['num_critic_for_reviews']=movies['num_critic_for_reviews'].fillna(0.0).astype(np.float32)
movies['director_facebook_likes']=movies['director_facebook_likes'].fillna(0.0).astype(np.float32)
movies['actor_3_facebook_likes'] = movies['actor_3_facebook_likes'].fillna(0.0).astype(np.float32)
movies['actor_1_facebook_likes'] = movies['actor_1_facebook_likes'].fillna(0.0).astype(np.float32)
movies['gross'] = movies['gross'].fillna(0.0).astype(np.float32)
movies['num_voted_users'] = movies['num_voted_users'].fillna(0.0).astype(np.float32)
movies['cast_total_facebook_likes'] = movies['cast_total_facebook_likes'].fillna(0.0).astype(np.float32)
movies['num_user_for_reviews'] = movies['num_user_for_reviews'].fillna(0.0).astype(np.float32)
movies['facenumber_in_poster'] = movies['facenumber_in_poster'].fillna(0.0).astype(np.float32)
movies['actor_2_facebook_likes'] = movies['actor_2_facebook_likes'].fillna(0.0).astype(np.float32)
movies['budget'] = movies['budget'].fillna(0.0).astype(np.float32)
movies['movie_facebook_likes'] = movies['movie_facebook_likes'].fillna(0.0).astype(np.float32)


movies['imdb_score']=movies['imdb_score'].fillna(0.0).astype(int)
# #movies.info()

my=list(zip(movies['director_facebook_likes'],movies['actor_1_facebook_likes'],movies['actor_2_facebook_likes'],movies['actor_3_facebook_likes']))
u = np.array(my)
u
X=u[:,:-1]
y=u[:,-1]
X
y
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)


clf=tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
print("DecisionTree",clf.score(X_test,y_test))

from sklearn.svm import SVC
clf=SVC(kernel="rbf")
clf.fit(X_train,y_train)
print("SVM",clf.score(X_test,y_test))

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
print("NaiveBayes",clf.score(X_test,y_test))

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
print("RandomForest",clf.score(X_test,y_test))

clf=AdaBoostClassifier()
clf.fit(X_train,y_train)
print("AdaBoost",clf.score(X_test,y_test))

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
print("KNeighbors",clf.score(X_test,y_test))