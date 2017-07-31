# coding=utf-8
import os
import re
import pickle
import pandas as pd

import nltk
# nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline

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

# Separando teste e treino
# 80% para treino e 20% para testes
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20, random_state=0)

# Utilizei pipeline para juntar os passos da transformação que podem ser validadas em conjunto
pipeline = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(norm='l2', use_idf=False, sublinear_tf=True),
        LogisticRegression(C=3.0, n_jobs=-2)
)


# Deixei os testes comentados só para documentação mesmo
# Classificador Regressão Logistica
#best_score = {'score': 0.00001, 'name': 1}
#print('Testando as configs ...')
#teste_ = 1.0
#while teste_ <= 10.0:
    #print(teste_)
    # classifier = LogisticRegression(C=teste_, n_jobs=-2, solver='lbfgs', multi_class='multinomial')
    #classifier = LogisticRegression(C=teste_, n_jobs=-2)
    #classifier.fit(X_train, y_train)
    # Testes
    #score = classifier.score(X_test, y_test)
    #print(score)
    #print('='*80)
    #if score > best_score['score']:
    #    best_score['score'] = score
    #    best_score['name'] = teste_
    #teste_ += 2

pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))# precisão 0.8515625

# Salva modelo
filename = 'song_recog_model.sav'
pickle.dump(pipeline, open(filename, 'wb'))
print('FIM')