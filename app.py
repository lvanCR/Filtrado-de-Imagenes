import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from filtros.suavizado import filtro_media_simple, filtro_media_ponderada, filtro_mediana
from filtros.agudizado import filtro_laplaciano, filtro_sobel
from utils.helpers import agregar_ruido_sal_pimienta, normalizar

# Carátula mejorada
st.markdown("<h1 style='text-align: center;'>Filtrado de Imágenes</h1>", unsafe_allow_html=True)
# Mostrar logo UPC centrado
import base64
logo_path = os.path.join("imagenes_ejemplo", "Upc.png")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"<div style='display: flex; justify-content: center;'><img src='data:image/png;base64,{encoded_string}' width='180'/></div>",
        unsafe_allow_html=True
    )
else:
    st.warning("No se encontró el logo en 'imagenes_ejemplo/Upc.png'")

st.markdown("<br>", unsafe_allow_html=True)  # Espacio arriba del logo
st.markdown("<div style='text-align: center; font-size: 1.2em; margin-bottom: 1em;'><b>Universidad Peruana de Ciencias Aplicadas (UPC)</b></div>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2.5px solid #003366; margin-top: 1.5em; margin-bottom: 1.5em;'>", unsafe_allow_html=True)

st.markdown("""
<b>Curso:</b> Matemática Computacional<br>
<b>Sección:</b> 6210<br>
<b>Docente:</b> Juan Manuel Mattos Quevedo<br>
<b>Grupo N°:</b> 3<br><br>
<b>Integrantes:</b><br>
- u202316818 - Cunyas Ramos Ivan<br>
- u20231c502 - Loza Condori, Ana Najhelyu<br>
- u202312723 - Ordaya Guerrero, Paolo<br>
- u202312506 - Melendez Arcos, Sergio<br>
- u202020107 - de Carvalho Saito, João Otavio<br>
<br>
<hr>
<div style='text-align: center;'>
<b>2025-10</b>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- INICIO DEL PROGRAMA PRINCIPAL DESDE LA LÍNEA 51 ---

# Paso 1: El usuario elige si sube una imagen o usa una de ejemplo
opcion_imagen = st.radio(
    "¿Qué imagen deseas usar?",
    ("Usar una imagen de ejemplo", "Subir mi propia imagen"),
)

# Paso 2: Selección de tipo de filtro
tipo_filtro = st.selectbox("Selecciona el Tipo de Filtro", ["Suavizado", "Agudizado"])

# Paso 3: Cargar imagen según la opción elegida
if opcion_imagen == "Subir mi propia imagen":
    uploaded_file = st.file_uploader("Sube tu propia imagen (JPG o PNG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        imagen_pil = Image.open(uploaded_file)
        st.markdown("**Imagen cargada por el usuario:**")
    else:
        st.info("Por favor, sube una imagen para continuar.")
        st.stop()
else:
    if tipo_filtro == "Suavizado":
        carpeta = os.path.join("imagenes_ejemplo", "suavizado")
    else:
        carpeta = os.path.join("imagenes_ejemplo", "agudizado")
    imagenes_ejemplo = [f for f in os.listdir(carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    opciones_sin_extension = [os.path.splitext(f)[0] for f in imagenes_ejemplo]
    opcion_seleccionada = st.selectbox("Elige una imagen de ejemplo", opciones_sin_extension)
    imagen_seleccionada = next((f for f in imagenes_ejemplo if os.path.splitext(f)[0] == opcion_seleccionada), None)
    if imagen_seleccionada:
        imagen_pil = Image.open(os.path.join(carpeta, imagen_seleccionada))
        st.markdown("**Imagen de ejemplo seleccionada:**")
    else:
        st.error("No se encontró la imagen seleccionada")
        st.stop()

# Convertir a escala de grises si es necesario
if imagen_pil.mode != "L":
    imagen_pil = imagen_pil.convert("L")
imagen_np = np.array(imagen_pil)

# Paso 4: Si es imagen de ejemplo y filtro de suavizado, aplicar ruido (solo una vez)
if opcion_imagen == "Usar una imagen de ejemplo" and tipo_filtro == "Suavizado":
    # Seleccionar tipo de ruido
    tipo_ruido= "suave"  # Por defecto, suave
    #tipo_ruido = st.selectbox(
     #   "Tipo de ruido sal y pimienta a aplicar",
      #  ["extremo", "suave"],
     #   help="Elige 'extremo' para ruido clásico (negro/blanco) o 'suave' para ruido menos agresivo."
    #)
    
    # Generar clave única para esta combinación imagen + tipo de ruido
    ruido_key = f"ruido_{imagen_seleccionada}_{tipo_ruido}"
    
    # Configurar parámetros de ruido
    if tipo_ruido == "extremo":
        intensidad = 0.12
        polarizacion = 40
    else:  # suave
        intensidad = 0.08
        polarizacion = 30
    
    # Si no tenemos esta combinación en session_state, aplicar ruido
    if ruido_key not in st.session_state:
        # Aplicar ruido y guardar en session_state
        imagen_ruido = agregar_ruido_sal_pimienta(
            imagen_np,
            intensidad=intensidad,
            tipo=tipo_ruido,
            polarizacion=polarizacion
        )
        st.session_state[ruido_key] = imagen_ruido
        st.session_state["ultima_imagen"] = imagen_seleccionada
        st.session_state["ultimo_tipo_ruido"] = tipo_ruido
    
    # Usar la imagen con ruido del session_state
    imagen_np = st.session_state[ruido_key]
    st.image(imagen_np, caption=f"Imagen con ruido sal y pimienta", use_container_width=True)
else:
    st.image(imagen_pil, caption="Imagen original", use_container_width=True)

# Aplicar filtro seleccionado
if tipo_filtro == "Suavizado":
    st.subheader("Selecciona el filtro de suavizado")
    filtro = st.radio("Tipo de filtro:", ["Media simple", "Media ponderada", "Mediana"], horizontal=True)

    # Estilo CSS para las tablas de máscaras de suavizado y su contenedor
    st.markdown("""
    <style>
    .mask-container-suavizado {
        display: flex;
        align-items: center; /* Centra verticalmente el texto y la tabla */
        justify-content: center; /* Centra horizontalmente todo el bloque */
        left: 0;
        gap: 10px; /* Espacio entre el texto y la tabla */
        width: fit-content; /* Ajustar el ancho al contenido */
    }
    .mask-table-suavizado {
        border-collapse: collapse;
        width: 120px; /* Ancho fijo para uniformidad */
    }
    .mask-table-suavizado td {
        border: 1px solid #777; /* Borde */
        padding: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        width: 40px; /* Ancho fijo para celdas */
        height: 40px; /* Altura fija para celdas */
        background-color: #f0f2f6; /* Color de fondo ligero */
    }
    .fraction-text {
        font-size: 1.8em; /* Tamaño más grande para la fracción */
        font-weight: bold;
    }
    .mask-description {
        text-align: center;
        font-size: 1.1em;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    resultado = None # Inicializar resultado

    if filtro == "Media simple":
        mask_simple_values = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        # Construir HTML para la máscara de media simple
        table_html_simple = "<table class='mask-table-suavizado'>"
        for row in mask_simple_values:
            table_html_simple += "<tr>"
            for val in row:
                table_html_simple += f"<td>{val}</td>"
            table_html_simple += "</tr>"
        table_html_simple += "</table>"
        
        # Combinar fracción y tabla en un contenedor flex
        st.markdown(f"""
        <div class='mask-container-suavizado'>
            <span class='fraction-text'>1/9</span>
            {table_html_simple}
        </div>
        """, unsafe_allow_html=True)
        resultado = filtro_media_simple(imagen_np, ksize=3)

    elif filtro == "Media ponderada":
        mask_ponderada_values = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        # Construir HTML para la máscara de media ponderada
        table_html_ponderada = "<table class='mask-table-suavizado'>"
        for row in mask_ponderada_values:
            table_html_ponderada += "<tr>"
            for val in row:
                table_html_ponderada += f"<td>{val}</td>"
            table_html_ponderada += "</tr>"
        table_html_ponderada += "</table>"
        
        # Combinar fracción y tabla en un contenedor flex
        st.markdown(f"""
        <div class='mask-container-suavizado'>
            <span class='fraction-text'>1/16</span>
            {table_html_ponderada}
        </div>
        """, unsafe_allow_html=True)
        resultado = filtro_media_ponderada(imagen_np)

    else: # Mediana
        resultado = filtro_mediana(imagen_np, ksize=3)
    
    if resultado is not None:
        st.image(resultado, caption="Imagen filtrada", use_container_width=True)

# El resto de tu código para el filtro de Agudizado permanece igual.

else: # Agudizado
    st.subheader("Selecciona el tipo de filtro de agudizado")
    tipo_agudizado = st.radio("Tipo de filtro:", ["Laplaciano", "Sobel"], horizontal=True)
    
    # Inicializar filtro_seleccionado en session_state si no existe
    if 'filtro_seleccionado' not in st.session_state:
        st.session_state.filtro_seleccionado = None
    
    # Estilo CSS global para todas las tablas
    st.markdown("""
    <style>
    .mask-table {
        left: 0;
        border-collapse: collapse;
        width: 120px; /* Ancho fijo para uniformidad */
    }
    .mask-table td {
    +    border: 1px solid #777; /* Borde, igual que suavizado */
        padding: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 16px; /* Tamaño de fuente, igual que suavizado */
        width: 40px; /* Ancho fijo para celdas, igual que suavizado */
        height: 40px; /* Altura fija para celdas, igual que suavizado */
        background-color: #f0f2f6; /* Color de fondo, igual que suavizado */
        }
    .mask-container {
        display: flex;
        flex-direction: column;
        align-items: center; /* Esto centra horizontalmente los elementos en la columna */
        margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if tipo_agudizado == "Laplaciano":
        st.markdown("**Máscaras Laplacianas disponibles:**")
        
        # Crear columnas para mostrar las máscaras
        col1, col2, col3 = st.columns(3)
        
        # Datos de las máscaras Laplacianas
        laplacian_masks = [
            {"title": "Clásico (4-vecinos)", "mask": [[0, 1, 0], [1, -4, 1], [0, 1, 0]], "key": "lap_clasico", "name": "Laplaciano clásico"},
            {"title": "8-conectado positivo", "mask": [[1, 1, 1], [1, -8, 1], [1, 1, 1]], "key": "lap_positivo", "name": "Laplaciano 8-conectado positivo"},
            {"title": "8-conectado negativo", "mask": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], "key": "lap_negativo", "name": "Laplaciano 8-conectado negativo"}
        ]

        for col, mask_info in zip([col1, col2, col3], laplacian_masks):
            with col:
                st.markdown(f"**{mask_info['title']}**")
                st.markdown("<div class='mask-container'>", unsafe_allow_html=True) # Contenedor para centrar

                table_html = "<table class='mask-table'>"
                for row in mask_info['mask']:
                    table_html += "<tr>"
                    for val in row:
                        table_html += f"<td>{val}</td>"
                    table_html += "</tr>"
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)

                if st.button(f"Usar {mask_info['title'].split(' ')[0]}", key=mask_info['key']):
                    st.session_state.filtro_seleccionado = mask_info['name']

                st.markdown("</div>", unsafe_allow_html=True) # Cerrar contenedor


    else:  # Sobel
        st.markdown("**Máscaras Sobel disponibles:**")
        
        # Crear columnas para mostrar las máscaras
        col1, col2 = st.columns(2)

        # Datos de las máscaras Sobel
        sobel_masks = [
            {"title": "Sobel X (horizontal)", "mask": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "key": "sobel_x", "name": "Sobel X"},
            {"title": "Sobel Y (vertical)", "mask": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "key": "sobel_y", "name": "Sobel Y"}
        ]

        for col, mask_info in zip([col1, col2], sobel_masks):
            with col:
                st.markdown(f"**{mask_info['title']}**")
                st.markdown("<div class='mask-container'>", unsafe_allow_html=True) # Contenedor para centrar

                table_html = "<table class='mask-table'>"
                for row in mask_info['mask']:
                    table_html += "<tr>"
                    for val in row:
                        # Resaltar valores negativos (opcional, para Sobel)
                            table_html += f"<td>{val}</td>"
                    table_html += "</tr>"
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)

                if st.button(f"Usar {mask_info['title'].split(' ')[0]}", key=mask_info['key']):
                    st.session_state.filtro_seleccionado = mask_info['name']

                st.markdown("</div>", unsafe_allow_html=True) # Cerrar contenedor

    # Aplicar el filtro seleccionado
    if st.session_state.filtro_seleccionado:
        st.markdown(f"**Filtro seleccionado:** {st.session_state.filtro_seleccionado}")
        
        if "Laplaciano" in st.session_state.filtro_seleccionado:
            if "clásico" in st.session_state.filtro_seleccionado:
                resultado = filtro_laplaciano(imagen_np, tipo='clasico')
            elif "positivo" in st.session_state.filtro_seleccionado:
                resultado = filtro_laplaciano(imagen_np, tipo='8-conectado-positivo')
            else:
                resultado = filtro_laplaciano(imagen_np, tipo='8-conectado-negativo')
        else:
            if "X" in st.session_state.filtro_seleccionado:
                resultado = filtro_sobel(imagen_np, direccion='x')
            else:
                resultado = filtro_sobel(imagen_np, direccion='y')
        
        resultado_normalizado = normalizar(resultado)
        st.image(resultado_normalizado, caption="Bordes detectados", use_container_width=True)