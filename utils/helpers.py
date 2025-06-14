import numpy as np
import base64
from io import BytesIO
from PIL import Image

def get_image_download_link(img_array, filename, text="Descargar imagen"):
    # Convertir el array de la imagen a un objeto de imagen PIL
    img = Image.fromarray(img_array)
    # Guardar la imagen en un buffer de bytes
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    # Codificar los bytes en base64
    b64 = base64.b64encode(img_bytes).decode()
    # Crear el enlace de descarga
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def agregar_ruido_sal_pimienta(
    imagen, 
    intensidad=0.08,  
    tipo="suave", 
    polarizacion=30, 
    rango_oscuro=(70, 120), 
    rango_claro=(130, 180)    
):
    salida = imagen.copy()
    x, y = salida.shape
    total_pixeles = x * y
    pixeles_cambiar = int(intensidad * total_pixeles)
    indices = np.random.choice(total_pixeles, pixeles_cambiar, replace=False)
    coords = np.unravel_index(indices, (x, y))
    
    for i in range(pixeles_cambiar):
        if tipo == "extremo":
            if np.random.randint(0, 2) == 0:
                salida[coords[0][i], coords[1][i]] = np.random.randint(0, polarizacion + 1)
            else:
                salida[coords[0][i], coords[1][i]] = np.random.randint(255 - polarizacion, 256)
        elif tipo == "suave":
            if np.random.randint(0, 2) == 0:
                salida[coords[0][i], coords[1][i]] = np.random.randint(rango_oscuro[0], rango_oscuro[1]+1)
            else:
                salida[coords[0][i], coords[1][i]] = np.random.randint(rango_claro[0], rango_claro[1]+1)
    return salida

def normalizar(imagen_np):
    minimo = np.min(imagen_np)
    maximo = np.max(imagen_np)
    if maximo - minimo == 0:
        return imagen_np.copy()
    norm = ((imagen_np - minimo) * (255.0 / (maximo - minimo))).astype(np.uint8)
    return norm

def get_image_data_for_histogram(image_array):
    """
    Obtiene los datos aplanados de una imagen y sus estadísticas para el histograma.
    Convierte el array a float64 para cálculos precisos, reflejando el rango actual de la imagen.
    """
    data_for_hist = image_array.ravel().astype(np.float64)
    min_val = np.min(data_for_hist)
    max_val = np.max(data_for_hist)
    mean_val = np.mean(data_for_hist)
    std_val = np.std(data_for_hist)
    return data_for_hist, min_val, max_val, mean_val, std_val