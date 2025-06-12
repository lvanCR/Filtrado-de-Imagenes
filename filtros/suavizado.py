import numpy as np
import cv2

def filtro_media_simple(imagen, ksize=3):
    """
    Aplica el filtro de media simple (máscara 3x3 de unos, normalizada por 1/9).
    """
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(imagen, -1, kernel)

def filtro_media_ponderada(imagen):
    """
    Aplica el filtro de media ponderada (máscara tipo Gaussiana 3x3, normalizada por 1/16).
    """
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], np.float32) / 16
    return cv2.filter2D(imagen, -1, kernel)

def filtro_mediana(imagen, ksize=3):
    """
    Aplica el filtro de la mediana (ventana 3x3).
    """
    return cv2.medianBlur(imagen, 3)