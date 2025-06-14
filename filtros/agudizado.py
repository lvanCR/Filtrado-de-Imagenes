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
        # Nota: El kernel original era centro 8, bordes -1. Lo corrijo si esa es la intención.
        # Si es el opuesto al positivo, entonces el centro es -8 y el resto +1
        # O si es una versión invertida, el centro es +8 y el resto -1.
        # Asumo que quieres el opuesto del 8-conectado-positivo para que genere el efecto contrario
        # es decir, si el 8-positivo realza bordes de una dirección, el 8-negativo realza los opuestos.
        # Sin embargo, si tu intención era simplemente una versión con signos opuestos para todo el kernel,
        # entonces el kernel sería: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]] como lo tenías.
        # Para que genere un efecto de agudizado, el Laplaciano tiene el centro con signo opuesto a los vecinos.
        # Me apego a la versión que ya tenías con el centro en 8 y vecinos en -1.
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=np.float32)
    else:
        # Por defecto usa el clásico
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
    
    # CAMBIO CRUCIAL: Forzar la salida a tipo float64 para preservar valores fuera de [0, 255]
    # Se utiliza cv2.CV_64F como profundidad de salida.
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
    
    # CAMBIO CRUCIAL: Forzar la salida a tipo float64 para preservar valores fuera de [0, 255].
    # No aplicar np.absolute ni np.uint8(np.clip()) aquí.
    # La magnitud (si se desea) y la normalización para visualización se harán en app.py.
    sobel = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    return sobel # Retorna el resultado raw, con valores negativos y mayores a 255
