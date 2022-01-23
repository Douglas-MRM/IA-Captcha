from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.layers.core import Flatten, Dense
from imutils.paths import list_images
from helper import redimensionar_img
from keras.models import Sequential
from os import sep
import numpy as np
import cv2 as cv
import pickle


'''
Iremos organizar as imagens e as respostas dos captchas
da pasta bdletras. Para que ao criar a Rede Neural da
IA, possamos treina-la com estes dados.

    NOTE: RNIA(Rede Neural da Inteligência Artificial)
'''

imagens = []
letras = []

for arquivo in list_images('bdletras'):
    # rotulo = bdletras/{A}/captcha1.png
    rotulo = arquivo.split(sep)[-2]
    imagem = cv.imread(arquivo)
    imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    
    # redimensiona a imagem para um tamanho de 20x20
    imagem = redimensionar_img(imagem, 20, 20)

    # adiciona uma terceira dimensão na imagem para o Keras poder
    # ler a imagem. Dimensão final: [axis=0, axis=1, {axis=2}]
    imagem = np.expand_dims(imagem, axis=2)

    # adicionando as informações as lista de imagens e letras
    letras.append(rotulo)
    imagens.append(imagem)


# A RNIA trabalha melhor com imagens que possuem uma intensidade de cores do
# pixel entre 0 e 1. No momento nossas imagens estão com essa itensidade
# entre 0 a 255, então iremos padronizar divindo por 255
imagens = np.array(imagens, dtype="float") / 255
letras  = np.array(letras)

# NOTE: separação entre imagens de treino(75%) e teste(25%).
# Isto é feito para evitar o problema de Overfitting(quando
# a RNIA não consegue achar uma resposta para novas imagens)
(X_train, X_test, y_train, y_test) = train_test_split(imagens, letras, test_size=0.25, random_state=0)

# NOTE: as nossas resposta são letras, porém a RNIA realiza cálculos
# matemáticos, então vamos converter as letras para números
lb = LabelBinarizer().fit(y_train)
y_train = lb.transform(y_train)
y_test  = lb.transform(y_test)


# NOTE: agora iremos salvar essas respostas convertidas em um aquivo, para
# consumi-los durante a resolução do captcha na arquivo resolver_captcha.py
with open('respostas-modelo.dat', 'wb') as file_pickle:
    pickle.dump(lb, file_pickle)

# NOTE: processo de criação da Rede Neural
# https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
modelo = Sequential()

# 1º camada
modelo.add(Conv2D(20, (5, 5), padding='same', input_shape=(20, 20, 1), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 2º camada
modelo.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 3º camada
modelo.add(Flatten())
modelo.add(Dense(500, activation='relu'))

# camada da saída
modelo.add(Dense(26, activation='softmax'))

# compilação de todas as camadas
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# trinar a RNIA com os dados de teste
modelo.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=26, epochs=10, verbose=1)

# salvar o modelo treinado em um arquivo
modelo.save('modelo_treinado.hdf5')
