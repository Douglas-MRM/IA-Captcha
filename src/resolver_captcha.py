from imutils.paths import list_images
from helper import redimensionar_img
from keras.models import load_model
from tratar_imagens import tratar
import numpy as np
import cv2 as cv
import pickle


def quebrar_captcha():
    '''
    Função responsável por tratar e solucionar
    os captchas contidos na pasta resolver.
    '''

    # importa o modelo que iremos utilizar para desconverter o resultado da IA
    with open('respostas-modelo.dat', 'rb') as arquivo_tradutor:
        lb = pickle.load(arquivo_tradutor)
    
    # importa o modelo treinado que sera usado para resolver os captchas
    modelo = load_model('modelo_treinado.hdf5')

    # realizamos o tratamento das imagens que iremos resolver
    tratar(pasta_origem='resolver', pasta_destino='resolver')

    for arquivo in list_images('resolver'):
        imagem = cv.imread(arquivo)
        imagem = cv.cvtColor(imagem, cv.COLOR_RGB2GRAY)

        # encontrar os contornos de cada letra
        contornos, _ = cv.findContours(imagem, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contornos_letras = []

        # filtra e guarda a localização de cada letra identificada
        for contorno in contornos:
            area = cv.contourArea(contorno)

            if area > 115:
                (x, y, largura, altura) = cv.boundingRect(contorno)
                contornos_letras.append((x, y, largura, altura))

        # ordena os contornos com base no eixo X(do menor valor no eixo
        # X para o maior) para seguir a ordem das letras no captcha
        contornos_letras = sorted(contornos_letras, key=lambda lista: lista[0])
        previsao = []

        # para cada letra iremos manda-lá para a IA solucionar
        for letra in contornos_letras:
            x, y, largura, altura = letra
            imagem_letra = imagem[y-2:y + largura +2 , x-2:x + largura + 2]

            # redimensiona a imagem para um tamanho de 20x20
            imagem_letra = redimensionar_img(imagem_letra, 20, 20)

            # para resolver o captcha o Keras precisa da imagem com
            # 4 dimensões Ex: [{axis=0}, 0-255, 0-255, {axis=2}]
            imagem_letra = np.expand_dims(imagem_letra, axis=2)
            imagem_letra = np.expand_dims(imagem_letra, axis=0)
            
            # mandamos essa imagem para o Modelo da IA resolver
            letra_prevista = modelo.predict(imagem_letra)
            # desconvertemos o resultado da IA(pois as letras foram convertidas para números)
            letra_prevista = lb.inverse_transform(letra_prevista)[0]
            # salvamos a letra encontrada
            previsao.append(letra_prevista)

        # converte a lista das letras encontradas em uma string
        resposta = ''.join(previsao)

        # return resposta do captcha
        print('>>>>>>>>>> ', resposta, ' <<<<<<<<<<')

if __name__ == '__main__':
    quebrar_captcha()
