from os.path import basename
from itertools import count
from glob import glob
import cv2 as cv


'''
Responsável por encontrar e traçar na imagem os contornos de letras identificadas.
E por fim, salva as letras na pasta 'idenficacao/letras' para que possamos popular
nossa pasta bdletras.
'''
for arquivo in glob('bdcaptchas/*'):
    if 'INFO.md' in arquivo:
            continue

    imagem = cv.imread(arquivo)
    imagem = cv.cvtColor(imagem, cv.COLOR_RGB2GRAY)

    # função para encontrar todos os contornos na imagem
    contornos_identificados, _ = cv.findContours(imagem, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # percorre cada contorno encontrado e adiciona na lista contornos_letras caso
    # o mesmo tenha uma area > 115(se informado uma area menor, ele pode passar a
    # identificar contornos que não são de letras)
    contornos_letras = []
    for contorno in contornos_identificados:
        area = cv.contourArea(contorno)

        if area < 116:
            continue

        (x, y, largura, altura) = cv.boundingRect(contorno)
        contornos_letras.append((x, y, largura, altura))


    # NOTE: caso ele não encontre as 5 letras na imagem eu pulo essa imagem(opcional)
    if len(contornos_letras) != 5:
        print('Pulando...' + arquivo)
        continue

    # nossa imagem esta apenas em uma escala de preto(0-255), porém para salva-lá
    # precisamos que a mesma esteja no formato RGB(0-255, 0-255, 0-255) então
    # multiplicamos uma lista com seu valor por três([0-255, 0-255, 0-255])
    imagem_final = cv.merge([imagem] * 3)
    i = count(start=1)


    # agora que temos as instruções(y, x, altura e largura) dos contornos
    # das letras identificadas, iremos cortar e salvar cada uma na pasta
    # identificacao/letras e salvar a imagem com os contornos traçados
    # na pasta identificacao/contornos
    for retangulo in contornos_letras:
        x, y, largura, altura = retangulo
        imagem_letra = imagem[y-3:y + altura +3 , x-3:x + largura + 3]
        
        nome_arquivo = basename(arquivo).replace('.png', f'_letra{next(i)}.png')
        cv.imwrite(f'identificacao/letras/{nome_arquivo}', imagem_letra)
        cv.rectangle(imagem_final, (x-3, y-3), (x+largura+3, y+altura+3), (255, 0, 0), 1)

    nome_arquivo = basename(arquivo)
    cv.imwrite(f'identificacao/contornos/{nome_arquivo}', imagem_final)
