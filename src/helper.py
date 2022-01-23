import imutils
import cv2


def redimensionar_img(image, largura: int, altura: int):
    '''
    Redimensiona a imagem para determinado tamanho informado
        :param image: imagem a ser redimensionada
        :param largura: largura desejada em pixels
        :param altura: altura desejada em pixels
    '''

    # pegue as dimensões da imagem e inicialize
    # os valores de preenchimento
    (h, w) = image.shape[:2]

    # se a largura for maior do que a altura, então
    # redimensionar ao longo da largura
    if w > h:
        image = imutils.resize(image, width=largura)

    # caso contrário, a altura é maior do que a largura,
    # por isso redimensiona ao longo da altura
    else:
        image = imutils.resize(image, height=altura)

    # determinar os valores de estofo para a largura
    # e  altura para obter as dimensões alvo
    padW = int((largura - image.shape[1]) / 2.0)
    padH = int((altura - image.shape[0]) / 2.0)

    # aplicar mais um redimensionamento para lidar
    # com quaisquer problemas de arredondamento
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (largura, altura))

    # devolver a imagem pré-processada
    return image
