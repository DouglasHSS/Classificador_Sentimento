# -*- coding: utf-8 -*-

from math import log
from nltk.classify import NaiveBayesClassifier, accuracy
from random import shuffle
from utils import tokenizar


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

    def __init__(self, dict_arquivos):
        """
        :param dict_arquivos: dicionario contendo os arquivos separados por classes.
        """
        self._inicializar_variaveis(dict_arquivos)
        self._calcular_idf(dict_arquivos)

        self._treinar_classificador()

    # #####################
    # # MÉTODOS PRIVADOS ##
    # #####################

    def _inicializar_variaveis(self, dict_arquivos):
        """Método para inicialização de elementos.
           Será criado a bag of words do classificador, bem como conjunto de dados utilizado no
           treinamento do classificador.

            :param dict_arquivos:  dicionario contendo os arquivos separados por classes.

            :return :None:
        """
        palavras = {}
        dados_treinamento = {}

        for classe, arquivos in dict_arquivos.items():

            dados_treinamento[classe] = []

            for caminho_arquivo in arquivos:
                arquivo = open(caminho_arquivo, mode="r")
                texto_arquivo = " ".join(arquivo.readlines()).decode("utf-8")

                tokens = tokenizar(texto_arquivo)

                for palavra in tokens.keys():
                    try:
                        palavras[palavra].add_ocorrencia()
                    except KeyError:
                        palavras[palavra] = Palavra(palavra=palavra)

                dados_treinamento[classe].append(tokens)

        self.bag_of_words = palavras.values()
        self.conjunto_treinamento = dados_treinamento

    def _calcular_idf(self, dict_arquivos):
        """Método que calcula o idf de cada palavra na bag word.
            :param dict_arquivos: dicionario contendo os arquivos separados por classes.

            :return :None:
        """
        total_arquivos = sum(len(arquivos)
                             for arquivos in dict_arquivos.values())

        for palavra in self.bag_of_words:
            palavra.calcular_idf(total_arquivos)

    def _treinar_classificador(self):
        """Método que treina o classificador

            :return: :None:
        """
        lista_treinamento = []

        for classe, lista_textos in self.conjunto_treinamento.items():
            for textos_tokenizados in lista_textos:
                caracteristicas = self._extrair_caracteristicas(textos_tokenizados)
                lista_treinamento.append((caracteristicas, classe))

        shuffle(lista_treinamento)

        self.classificador = NaiveBayesClassifier.train(lista_treinamento)
    
    def _extrair_caracteristicas(self, tokens):
        """Método que extrai as caracteristicas de uma lista tokens de um review.
            :param tokens: :dict: key:token value:tf*idf

            :return :dict: com as caracteristicas extraidas
        """
        numero_tokens = sum(tokens.values())

        def calcular_tf(palavra_obj):
            frequencia_token = tokens.get(palavra_obj.palavra, 0)
            return float(frequencia_token)/numero_tokens

        return {palavra.palavra: calcular_tf(palavra) * palavra.idf
                for palavra in self.bag_of_words}

    # #####################
    # # MÉTODOS PÚBLICOS ##
    # #####################

    def classificar_reviews(self, arquivos):
        """ Método que classifica uma lista de arquivos.
            :param arquivos: :list: dos caminhos dos arquivos a serem classificados.

            :return: :list: contendo a classificação dos respectivos arquivos.
        """
        lista_de_caracteristicas = []
        for caminho_arquivo in arquivos:
            arquivo = open(caminho_arquivo, mode="r")
            texto_arquivo = " ".join(arquivo.readlines()).decode("utf-8")
            tokens = tokenizar(texto_arquivo)

            lista_de_caracteristicas.append((self._extrair_caracteristicas(tokens)))

        return self.classificador.classify_many(lista_de_caracteristicas)

    def medir_taxa_acerto(self, conjunto_de_teste):
        """Método para medir a taxa de acerto do classificador.
            :param conjunto_de_teste: :list: contendo :tuple:(:dict: de caracteristicas, classe)

            :return: :float:
        """
        return accuracy(self.classificador, conjunto_de_teste)