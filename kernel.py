#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib.colors import ListedColormap

movies = pd.read_csv('data/movie_metadata.csv')

#print(movies.info())
print(movies.color.describe());
#print(movies.color.mode()[0]);
#print(movies.dtypes)

movies.color = movies.color.fillna(movies.color.mode()[0])
movies.color = movies.color.map({'Color': 1, ' Black and White':0})

movies = movies.fillna(0)


print(movies.color.describe());
total_attributes = movies.shape[1] - 1

y = movies.imdb_score.shift(0)
X = movies.drop('imdb_score', axis=1)


X = X.drop('director_name', axis=1)
X = X.drop('actor_1_name', axis=1)
X = X.drop('actor_2_name', axis=1)
X = X.drop('actor_3_name', axis=1)
X = X.drop('genres', axis=1)
X = X.drop('movie_title', axis=1)
X = X.drop('plot_keywords', axis=1)
X = X.drop('movie_imdb_link', axis=1)
X = X.drop('language', axis=1)
X = X.drop('country', axis=1)
X = X.drop('content_rating', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


#f, ax = plt.subplots(figsize=(12, 10))
#plt.title('Correlation of Movie Features')
# Draw the heatmap using seaborn
#sns.heatmap(movies.corr(),linewidths=0.1, square=True, cmap="YlGnBu", linecolor='black', annot=True)

# Plot de ocorrencia de cada valor, por coluna
#movies.hist(figsize=(14, 26))
#plt.show()

# movies = movies.drop('actor_1_facebook_likes', axis=1)

#vect = DictVectorizer(sparse=False)
#X_dict = movies.director_name.to_dict().values()
#X_train = vect.fit_transform(X_dict)

#lnr = GaussianNB().fit(X_train, y_train)
lnr = LinearRegression().fit(X_train, y_train)
#lnr = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)

y_prev = lnr.predict(X_test)

#predictions = X_test
#predictions["nota_real"] = y_test
#predictions["nova_estimada"] = y_predict


plt.figure(figsize=(10, 10))
cm2 = ListedColormap(['#0000aa', '#ff6030'])

#plt.ylim(0, 10)
#plt.plot(predictions.nota_real[0:500],'o', c=cm2(0))
#plt.plot(predictions.nova_estimada[0:500],'o', c=cm2(1))
#plt.grid(True)
#plt.title("Erro de previsão")

#plt.ylim(0, 10)
plt.plot(y_prev-y_test,'o', c=cm2(0))
#plt.plot(a[0:500],'o', c=cm2(1))
plt.grid(True)
plt.title("Erro de previsão")




# Calcula o erro absoluto e o erro percentual da regressao linear
errolnr = np.abs(y_test - y_prev)
erroperc = errolnr / list(y_test)

print('Erro percentual:\n Média: {0:.2f}  Max: {1:.2f}   Min: {2:.2f}'
      .format(np.mean(erroperc), np.max(erroperc), np.min(erroperc)))

plt.figure(figsize=(6, 4))
plt.errorbar(np.arange(y_test.size), list(y_test), yerr=errolnr,
             fmt='.', ecolor='r', capsize=3)
plt.title("Valores reais (barras de erro de predição)")
plt.grid()

fig, ax = plt.subplots(1, 2, figsize=(7, 4))
plt.suptitle("Erros de previsão")
ax[0].plot(errolnr,'.')
ax[0].set_xlabel("Erro absoluto")
ax[0].grid()
ax[1].plot(erroperc,'.')
ax[1].set_xlabel("Erro percentual")
ax[1].grid()


   

#print("\nRegressao Linear")
print("Acurácia da base de treinamento: {:.2f}".format(lnr.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(lnr.score(X_test, y_test)))
#print("w: {}  b: {}".format(lnr.coef_, lnr.intercept_))
