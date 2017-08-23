# coding=utf-8
import os
import re
import pickle
import pandas as pd
from random import uniform
import nltk
# nltk.download('stopwords')

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

stopwords_ = nltk.corpus.stopwords.words('portuguese')
regex = re.compile('[^a-zA-Z]')
repetitions_regex = re.compile('(\[x\d+\])')
repetitions_regex2 = re.compile('(\(x\d+\))')
path = os.getcwd() + '/files/lyrics/'


def clean_data(text):
    result = repetitions_regex.findall(text)
    if result:
        for r in result:
            text = text.replace(r, ' ')
    result = repetitions_regex2.findall(text)
    if result:
        for r in result:
            text = text.replace(r, ' ')
    return text.replace('(', '').replace(')', '')

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
    teste = teste.replace('\n', ' ').replace(',', ' ').replace('.', ' ')
    teste = clean_data(teste)
    teste = ' '.join([x for x in teste.split() if x not in set(stopwords_)])
    corpus.append(teste)

# Separando teste e treino
# 80% para treino e 20% para testes
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20, random_state=0)

best_score = 0.000000000001
epochs = 500
for x in range(epochs):
    c = uniform(0, 50)
    # Utilizei pipeline para juntar os passos da transformação que podem ser validadas em conjunto
    pipeline = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(norm='l2', use_idf=False, sublinear_tf=True),
        LogisticRegression(C=c, n_jobs=-2)#, multi_class='multinomial', solver='lbfgs')
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    if score > best_score:
        print('Epoch {0} - C = {1} score {2}'.format(x, c, str(score)))
        print('Best one epoch {0} with C = {1}!!!!!!! Saved.'.format(x, c))
        best_score = score
        # Salva modelo
        filename = 'song_recog_model.sav'
        pickle.dump(pipeline, open(filename, 'wb'))# precisão 0.8515625
    print('-' * 80)

# y_pred = pipeline.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
print('FIM')

