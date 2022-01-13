from os.path import basename
from glob import glob
from PIL import Image
import numpy as np
import cv2 as cv


def tratar(pasta_origem, pasta_destino='bdcaptchas'):
    '''
    Realiza o mesmo procedimento explicado no arquivo metodos_cv2.py(basicamente
    aplica o método cv.THRESH_TRUNC na imagem, e depois a transforma para preto
    e branco) porém aqui o processo é realizado para várias imagens.
        :param pasta_origem: pasta onde estão as imagens
        :param pasta_destino: pasta de salvamento das imagens processadas
    '''

    for arquivo in glob(f'{pasta_origem}/*'):
        if 'INFO.md' in arquivo:
            continue

        imagem = cv.imread(arquivo)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(imagem, cv.MORPH_OPEN, kernel)

        imagem_cinza = cv.cvtColor(opening, cv.COLOR_RGB2GRAY)

        _, imagen_tratada = cv.threshold(imagem_cinza, 127, 255, cv.THRESH_TRUNC or cv.THRESH_OTSU)
        cv.imwrite(f'{pasta_destino}/{basename(arquivo)}', imagen_tratada)

        imagem = Image.open(f'{pasta_destino}/{basename(arquivo)}')
        imagem = imagem.convert('P')

        imagem_final = Image.new('P', imagem.size, (0, 0, 0))

        for x in range(imagem.size[1]):
            for y in range(imagem.size[0]):
                cor_pixel = imagem.getpixel((y, x))

                if cor_pixel < 115:
                    imagem_final.putpixel((y, x), (255, 255, 255))

        imagem_final.save(f'{pasta_destino}/{basename(arquivo)}')


if __name__ == '__main__':
    tratar('bdcaptchas')
