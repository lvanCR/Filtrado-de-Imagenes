import numpy as np

def agregar_ruido_sal_pimienta(
    imagen, 
    intensidad=0.08,  # Intensidad reducida para ruido suave
    tipo="suave", 
    polarizacion=30,  # Polarización reducida
    rango_oscuro=(70, 120),   # Rango medio-bajo
    rango_claro=(130, 180)    # Rango medio-alto
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

def diferencia(imagen1, imagen2):
    dif = np.abs(imagen1.astype(np.int16) - imagen2.astype(np.int16))
    dif = np.clip(dif, 0, 255).astype(np.uint8)
    return dif