#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#import numpy as np 
import pandas as pd 

movies = pd.read_csv('data/movie_metadata.csv')


#print(movies.info())
#print(movies.color.describe());
#print(movies.color.mode()[0]);
#print(movies.dtypes)

movies.color = movies.color.fillna(movies.color.mode()[0])
movies.color = movies.color.map({'Color': 1, ' Black and White':0})


print(movies.color.describe());


