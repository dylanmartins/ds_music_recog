# coding=utf-8
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
#nltk.download('stopwords')
from unicodedata import normalize

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

stopwords_ = nltk.corpus.stopwords.words('portuguese') + [
    u'r$', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9',
]
regex = re.compile('[^a-zA-Z]')
path = os.getcwd() + '/files/lyrics/'


def remove_accents(input_str):
    """
    :param input_str: String com acentos para serem removidos
    :return:
    """
    text = normalize('NFKD', input_str).encode('ASCII', 'ignore').decode('ASCII')
    # retirar simbolos especiais
    text = re.sub('[]')
    return


def div_corpus(corpus, labels):
    # 20% para testes / 80% treinos
    percent_div = int(len(corpus)*0.8)

    # Treinos
    train_x = corpus[:percent_div]
    train_y = labels[:percent_div]

    # Testes
    test_x = corpus[percent_div:]
    test_y = labels[percent_div:]

    return train_x, train_y, test_x, test_y

dados, labels = [], []
print('Pegando os dados ...')
for csv in os.listdir(path):
    # Pegando os dados
    label = csv.split('.csv')[0].upper()
    data = pd.read_csv(path+csv)
    for x in data['lyric']:
        dados.append(x)
        labels.append(label)

# Divide treino e teste
train_x, train_y, test_x, test_y = div_corpus(dados, labels)

corpus = []
print('Limpando dados ...')
for dado in train_x:
    # Limpando dados
    teste = regex.sub(dado, ' ').lower()
    teste = ' '.join([remove_accents(x) for x in teste.split() if x not in set(stopwords_)])
    corpus.append(teste)

# Bag of words
print('Criando bag of words ...')
X = cv.fit_transform(corpus).toarray()
print('FIM ' + label)

print('eita')