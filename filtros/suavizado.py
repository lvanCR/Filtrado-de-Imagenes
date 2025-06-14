import numpy as np
import cv2

def filtro_media_simple(imagen, ksize=3):
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(imagen, -1, kernel)

def filtro_media_ponderada(imagen):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], np.float32) / 16
    return cv2.filter2D(imagen, -1, kernel)

def filtro_mediana(imagen, ksize=3):
    return cv2.medianBlur(imagen, 3)