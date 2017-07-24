# -*- coding: utf-8 -*-
import string

from collections import Counter
from nltk.corpus import stopwords
from unicodedata import normalize


STOPWORDS = set(stopwords.words('english'))
PONTUACAO = set(string.punctuation)


def normalizar_texto(texto):
    """Função que substitui caracteres especiais por caracteres normais.
        :param texto: :unicode:

        :return :unicode:
    """
    return normalize('NFKD', texto).lower()


def tokenizar(texto):
    """Função de tokenização de um texto. Stopwords e Pontuações serão removidas.
        :param texto: :unicode:

        :return :dict: key:token value:frequencia
    """
    tokens = Counter(normalizar_texto(texto).split())

    for palavra in STOPWORDS.union(PONTUACAO):
        if palavra in tokens:
            tokens.pop(palavra)

    return tokens
