# -*- coding: utf-8 -*-
import os
from classificador import ClassificadorSentimento


def caso_de_teste():
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


if __name__ == "__main__":
    caso_de_teste()
