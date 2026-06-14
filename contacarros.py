import cv2
import numpy as np

arquivo_video = cv2.VideoCapture('contarcarros.mp4')

total_veiculos = 0
pronto_para_contar = False

detector_movimento = cv2.createBackgroundSubtractorMOG2(
    history=120,
    varThreshold=35,
    detectShadows=False
)

while True:

    sucesso, frame = arquivo_video.read()

    if not sucesso:
        break

    frame = cv2.resize(frame, (1100, 720))

    # Nova área de monitoramento
    pos_x = 250
    pos_y = 280
    largura = 350
    altura = 140

    mascara_movimento = detector_movimento.apply(frame)

    elemento = np.ones((5, 5), np.uint8)

    mascara_movimento = cv2.morphologyEx(
        mascara_movimento,
        cv2.MORPH_OPEN,
        elemento
    )

    imagem_tratada = cv2.dilate(
        mascara_movimento,
        elemento,
        iterations=2
    )

    area_observada = imagem_tratada[
        pos_y:pos_y + altura,
        pos_x:pos_x + largura
    ]

    pixels_movimento = cv2.countNonZero(area_observada)

    tecla = cv2.waitKey(20) & 0xFF

    if pixels_movimento > 4500 and pronto_para_contar:
        total_veiculos += 1
        pronto_para_contar = False

    if pixels_movimento < 2000:
        pronto_para_contar = True

    cor_area = (255, 0, 0) if pronto_para_contar else (0, 255, 255)

    cv2.rectangle(
        frame,
        (pos_x, pos_y),
        (pos_x + largura, pos_y + altura),
        cor_area,
        4
    )

    cv2.rectangle(
        imagem_tratada,
        (pos_x, pos_y),
        (pos_x + largura, pos_y + altura),
        255,
        3
    )

    cv2.putText(
        frame,
        f'Veiculos: {total_veiculos}',
        (400, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        4
    )

    cv2.imshow('Monitoramento', frame)

    if tecla == 27:
        break

arquivo_video.release()
cv2.destroyAllWindows()