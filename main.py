# -*- coding: utf-8 -*-
import string

from math import log
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier, accuracy
from random import shuffle
from unicodedata import normalize


STOPWORDS = stopwords.words('english')
PONTUACAO = string.punctuation


def normalizar_texto(texto):
    """Função que substitui caracteres especiais por caracteres normais.
        :param texto: :unicode:

        :return :unicode:
    """
    return normalize('NFKD', texto).lower()


def separar_tokens(texto):
    """Função que retornar os tokens. Stopwords e pontuações serão removidas.
        :param texto: :unicode:

        :return :set: de tokens
    """
    return (set(normalizar_texto(texto).split())
            .difference(STOPWORDS)
            .difference(PONTUACAO))


class Palavra(object):

    def __init__(self, palavra):
        self.palavra = palavra
        self.ocorrencia = 1
        self.idf = None

    def __repr__(self):
        return "Palavra '{0}'".format(self.palavra)

    def add_ocorrencia(self):
        """Método que incrementa em 1 quando a palavra ocorre em um documento.
            :return :None:
        """
        self.ocorrencia += 1

    def calcular_idf(self, total_documentos):
        """Método que calcula o idf de uma palavra.
            :param total_documentos: :int: referente ao total de documentos.

            :return :None:
        """
        self.idf = log(float(total_documentos) / float(self.ocorrencia))


class ClassificadorSentimento(object):

    # ###############
    # # CONSTRUTOR ##
    # ###############

    def __init__(self, reviews_positivos, reviews_negativos):
        """
        :param reviews_positivos: :list: dos arquivos positivos que treinarão o classificador.
        :param reviews_negativos: :list: dos arquivos negativos que treinarão o classificador.
        """

        self.conjunto_treinamento = {"positivo": reviews_positivos,
                                     "negativo": reviews_negativos}

        self.bow_positiva = self._criar_bow(reviews_positivos)
        self.bow_negativa = self._criar_bow(reviews_negativos)

        self._treinar_classificador()

    # #####################
    # # MÉTODOS PRIVADOS ##
    # #####################

    def _criar_bow(self, arquivos):
        """Method que cria uma bag of words de um corpus.
            :param arquivos: :list: dos caminhos dos arquivos do corpus.

            :return :list: de :Palavra:
        """
        dict_palavras = {}

        for caminho_arquivo in arquivos:
            arquivo = open(caminho_arquivo, mode="r")
            texto_arquivo = " ".join(arquivo.readlines()).decode("utf-8")

            for palavra in separar_tokens(texto_arquivo):
                try:
                    dict_palavras[palavra].add_ocorrencia()
                except KeyError:
                    dict_palavras[palavra] = Palavra(palavra=palavra)

        for palavra in dict_palavras.values():
            palavra.calcular_idf(len(arquivos))

        return dict_palavras.values()

    def _treinar_classificador(self):
        """ Método que treina o classificador

            :return: :None:
        """
        lista_treinamento = []

        for classe, arquivos in self.conjunto_treinamento.items():
            for caminho_arquivo in arquivos:
                arquivo = open(caminho_arquivo, mode="r")
                texto_arquivo = " ".join(arquivo.readlines()).decode("utf-8")

                lista_treinamento.append((self.extrair_caracteristicas(texto_arquivo), classe))

        shuffle(lista_treinamento)

        self.classificador = NaiveBayesClassifier.train(lista_treinamento)

    # #####################
    # # MÉTODOS PÚBLICOS ##
    # #####################

    def extrair_caracteristicas(self, texto):
        """Method que cria uma bag of words de um corpus.
            :param texto: :unicode: que terá as caracteristicas extraidas.

            :return :dict: com as caracteristicas extraidas
        """
        frequencia = FreqDist(texto.split())

        def calcular_tf(palavra_obj):
            return float(frequencia[palavra_obj.palavra])/len(texto)

        def somar_tf_idf(bag_of_words):
            return sum(calcular_tf(palavra_obj) * palavra_obj.idf
                       for palavra_obj in bag_of_words)

        return {"feature-positiva": somar_tf_idf(self.bow_positiva),
                "feature-negativa": somar_tf_idf(self.bow_negativa)}

    def classificar_reviews(self, arquivos):
        """ Método que classifica uma lista de arquivos.
            :param arquivos: :list: dos caminhos dos arquivos a serem classificados.

            :return: :list: contendo a classificação dos respectivos arquivos.
        """
        lista_de_caracteristicas = []
        for caminho_arquivo in arquivos:
            arquivo = open(caminho_arquivo, mode="r")
            texto_arquivo = " ".join(arquivo.readlines()).decode("utf-8")

            lista_de_caracteristicas.append((self.extrair_caracteristicas(texto_arquivo)))

        return self.classificador.classify_many(lista_de_caracteristicas)

    def medir_taxa_acerto(self, conjunto_de_teste):
        """Método para medir a taxa de acerto do classificador.
            :param conjunto_de_teste: :list: contendo :tuple:(:dict: de caracteristicas, classe)

            :return: :float:
        """
        return accuracy(self.classificador, conjunto_de_teste)


def caso_de_teste():
    import os
    print "Iniciando o teste..."
    reviews_positivos = [os.path.join("./reviews/pos", nome_arquivo)
                         for nome_arquivo in os.listdir("./reviews/pos")]

    reviews_negativos = [os.path.join("./reviews/neg", nome_arquivo)
                         for nome_arquivo in os.listdir("./reviews/neg")]

    print "Inicializando o classificador..."
    classificador = ClassificadorSentimento(reviews_positivos[:500],
                                            reviews_negativos[:500])

    conjunto_teste = reviews_negativos[900:] + reviews_positivos[900:]
    print "Classificando Reviews de teste..."
    print classificador.classificar_reviews(conjunto_teste)
    print "Teste Finalizado!"