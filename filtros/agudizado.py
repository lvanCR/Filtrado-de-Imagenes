import numpy as np
import cv2

def filtro_laplaciano(imagen, tipo='clasico'):
    if tipo == 'clasico' or tipo == '4-conectado':
        # Kernel clásico de Laplaciano 3x3 (4-conectado)
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
    elif tipo == '8-conectado-positivo':
        # Kernel Laplaciano 8-conectado (centro 8, bordes 1)
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], dtype=np.float32)
    elif tipo == '8-conectado-negativo':
        # Kernel Laplaciano 8-conectado (centro -1, bordes 8)
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=np.float32)
    else:
        # Por defecto usa el clásico
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
    
    return cv2.filter2D(imagen, cv2.CV_64F, kernel)

def filtro_sobel(imagen, direccion='x'):
    if direccion == 'x':
        # Sobel en X
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
    else:
        # Sobel en Y
        kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
    
    sobel = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    return sobel 
