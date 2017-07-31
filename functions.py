# coding=utf-8
import os
import re
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
regex = re.compile('[^a-zA-Z]')
stopwords_ = nltk.corpus.stopwords.words('portuguese')


def text_recog(text):
    # retornar resultado da IA
    filename = os.getcwd() + '/models/song_recog_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    corpus = []
    dados = [text]
    for dado in dados:
        # Limpando dados
        teste = regex.sub(dado, ' ').lower()
        teste = ' '.join([x for x in teste.split() if x not in set(stopwords_)])
        corpus.append(teste)

    # Bag of words
    teste = loaded_model.predict(corpus)
    return teste[0]