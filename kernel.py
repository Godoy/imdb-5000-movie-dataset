#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#import numpy as np 
import pandas as pd 


movies = pd.read_csv('data/movie_metadata.csv')

print(movies.describe());