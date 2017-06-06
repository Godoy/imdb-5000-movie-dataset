#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


movies = pd.read_csv('data/movie_metadata.csv')


#print(movies.info())
#print(movies.color.describe());
#print(movies.color.mode()[0]);
#print(movies.dtypes)

movies.color = movies.color.fillna(movies.color.mode()[0])
movies.color = movies.color.map({'Color': 1, ' Black and White':0})


print(movies.color.describe());


f, ax = plt.subplots(figsize=(12, 10))
plt.title('Correlation of Movie Features')
# Draw the heatmap using seaborn
sns.heatmap(movies.corr(),linewidths=0.1, square=True, cmap="YlGnBu", linecolor='black', annot=True)