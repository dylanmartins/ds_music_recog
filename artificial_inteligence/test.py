# coding=utf-8
import os
import re
import pandas as pd

import nltk
# nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=20000)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

stopwords_ = nltk.corpus.stopwords.words('portuguese')
regex = re.compile('[^a-zA-Z]')
path = os.getcwd() + '/files/lyrics/'

dados, y = [], []
print('Pegando os dados ...')
for csv in os.listdir(path):
    # Pegando os dados
    label = csv.split('.csv')[0].upper()
    data = pd.read_csv(path+csv)
    for x in data['lyric']:
        dados.append(x)
        y.append(label)

corpus = []
print('Limpando dados ...')
for dado in dados:
    # Limpando dados
    teste = regex.sub(dado, ' ').lower()
    teste = ' '.join([x for x in teste.split() if x not in set(stopwords_)])
    corpus.append(teste)

# Bag of words
print('Criando bag of words ...')
X = cv.fit_transform(corpus).toarray()

# Separando teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Classificador Regressão Logistica
best_score = {'score': 0.00001, 'name': 1, 'kernel': 'tey'}

print('Testando as configs ..')
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
for kn in kernels:
    teste_ = 1.0
    while teste_ <= 10.0:
        classifier = SVC(C=teste_, kernel=kn)
        # classifier = LogisticRegression(C=teste_, n_jobs=-2)
        classifier.fit(X_train, y_train)
        # Testes
        score = classifier.score(X_test, y_test)
        if score > best_score['score']:
            best_score['score'] = score
            best_score['name'] = teste_
            best_score['kernel'] = kn
        teste_ += 1

print('BEST {0} {1} SCORE {2}'.format(str(best_score['kernel']), str(best_score['name']), str(best_score['score'])))
print('FIM')