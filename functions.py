# coding=utf-8
import os
import re
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
regex = re.compile('[^a-zA-Z]')
stopwords_ = nltk.corpus.stopwords.words('portuguese')


def clean_data(text):
    repetitions_regex = re.compile('(\[x\d+\])')
    repetitions_regex2 = re.compile('(\(x\d+\))')
    result = repetitions_regex.findall(text)
    if result:
        for r in result:
            text = text.replace(r, ' ')
    result = repetitions_regex2.findall(text)
    if result:
        for r in result:
            text = text.replace(r, ' ')
    return text.replace('(', '').replace(')', '')


def text_recog(text):
    # retornar resultado da IA
    filename = os.getcwd() + '/models/song_recog_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    corpus = []
    dados = [text]
    for dado in dados:
        # Limpando dados
        teste = regex.sub(dado, ' ').lower()
        teste = teste.replace('\n', ' ').replace(',', ' ').replace('.', ' ')
        teste = clean_data(teste)
        teste = ' '.join([x for x in teste.split() if x not in set(stopwords_)])
        corpus.append(teste)

    # Bag of words
    teste = loaded_model.predict(corpus)
    proba = max(loaded_model.predict_proba(corpus)[0])
    return 'Estilo musical: <b>{0}</b></br>Probabilidade: <b>{1}</b>'.format(teste[0], proba)