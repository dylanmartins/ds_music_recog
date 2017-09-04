# coding=utf-8
import os
from collections import Counter
import numpy as np
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

stopwords_ = nltk.corpus.stopwords.words('portuguese') + [u'é', u'pra', u'amor', u'vai', u'vou', u'tá', u'vem',
                                                          u'hoje', u'to', u'quero', u'ai', u'gente', u'tô', u'sei']
regex = re.compile('[^a-zA-Z]')
repetitions_regex = re.compile('(\[x\d+\])')
repetitions_regex2 = re.compile('(\(x\d+\))')
path = os.getcwd() + '/files/lyrics/'

sertanejo_counts = Counter()
funk_counts = Counter()
gospel_counts = Counter()
bossa_counts = Counter()
total_counts = Counter()


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


dados, labels = [], []
print('Pegando os dados ...')
for csv in os.listdir(path):
    # Pegando os dados
    label = csv.split('.csv')[0].upper()
    data = pd.read_csv(path+csv)
    for x in data['lyric']:
        dados.append(x)
        labels.append(label)

corpus = []
print('Limpando dados ...')
for dado in dados:
    # Limpando dados
    teste = regex.sub(dado, ' ').lower()
    teste = teste.replace('\n', ' ').replace(',', ' ').replace('.', ' ')
    teste = clean_data(teste)
    teste = ' '.join([x for x in teste.split() if x not in set(stopwords_)])
    corpus.append(teste)

for i in range(len(corpus)):
    if labels[i] == 'SERTANEJO':
        for word in corpus[i].split(" "):
            sertanejo_counts[word] += 1
            total_counts[word] += 1
    elif labels[i] == 'FUNK':
        for word in corpus[i].split(" "):
            funk_counts[word] += 1
            total_counts[word] += 1
    elif labels[i] == 'GOSPEL':
        for word in corpus[i].split(" "):
            gospel_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in corpus[i].split(" "):
            bossa_counts[word] += 1
            total_counts[word] += 1

print('FIM')