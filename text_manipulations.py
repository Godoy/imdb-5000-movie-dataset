# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 23:39:22 2017

@author: Douglas
"""
import pandas as pd 
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction import DictVectorizer

movies = pd.read_csv('data/movie_metadata.csv')

#s = pd.Series([1, 2, 3, 1, np.nan])
#s2 = s.map('this is a string {}'.format, na_action='ignore')
#print(s2)

N = 10

X = movies.director_name.drop_duplicates()
keys = pd.Series(range(1, N + 1 ,1))

tamanho = X.count()
directors = pd.Series(X, index=range(1, tamanho + 1 ,1))

print(X)
print(directors)

X_df = pd.DataFrame({'director':X.unique()})
print(X_df)
#print(movies.columns)
##X_test = (['director_name' for i in X])
##print(X_test)

#X = X.fillna(X.mode()[0])

#vec = DictVectorizer()
#X_test = vec.fit(X_df)



