from itertools import count
from PIL import Image
import numpy as np
import cv2 as cv


imagem = cv.imread('bdcaptchas/captcha0.png')

if not isinstance(imagem, np.ndarray):
    exit('Não localizei a imagem para realizar o teste dos métodos.')

# reduz os ruídos da imagem
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(imagem, cv.MORPH_OPEN, kernel)

# transforma a imagem para tons de cinza
imagem_cinza = cv.cvtColor(opening, cv.COLOR_RGB2GRAY)

# metódos para tratamento de imagem que iremos testar
metodos = [
    cv.THRESH_BINARY,
    cv.THRESH_BINARY_INV,
    cv.THRESH_TRUNC,
    cv.THRESH_TOZERO,
    cv.THRESH_TOZERO_INV
]

i = count(start=1)

# criamos imagens aplicando cada um dos métodos determinados
for metodo in metodos:
    _, imagem_tratada = cv.threshold(imagem_cinza, 127, 255, metodo or cv.THRESH_OTSU)
    cv.imwrite(f'test/img_metodo_{next(i)}.png', imagem_tratada)

# pegamos a imagem que ficou com o melhor efeito(cv.THRESH_TRUNC) e
# convertemos para tons de cinza(precaução caso o PIL
# não identifique que a imagem esta em tons de cinza)
imagem = Image.open('test/img_metodo_3.png')
imagem = imagem.convert('P')

# precisamos de uma imagem em preto e branco. Então primeiramente
# criamos uma Imagem totalmente em branco
imagem_final = Image.new('P', imagem.size, (255, 255, 255))

# depois percorremos cada pixel da imagem que ficou com o melhor
# efeito(img_metodo_3.png) e verificamos se a tonalidade é menor
# que 115(um cinza mais escuro), se for 'pinte' na imagem_final(
# na mesma localização x e y) um pixel na cor preta
for x in range(imagem.size[1]):
    for y in range(imagem.size[0]): 
        cor_pixel = imagem.getpixel((y, x))

        if cor_pixel < 115:
            imagem_final.putpixel((y, x), (0, 0, 0))

# salvamos essa imagem_final que esta em preto e branco
imagem_final.save('test/imagem_final.png')
