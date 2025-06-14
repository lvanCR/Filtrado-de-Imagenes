import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from filtros.suavizado import filtro_media_simple, filtro_media_ponderada, filtro_mediana
from filtros.agudizado import filtro_laplaciano, filtro_sobel
from utils.helpers import agregar_ruido_sal_pimienta, normalizar, get_image_download_link, get_image_data_for_histogram
import base64
import matplotlib.pyplot as plt

st.markdown("<h1 style='text-align: center;'>Filtrado de Imágenes</h1>", unsafe_allow_html=True)

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

st.markdown("<br>", unsafe_allow_html=True) 
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

# --- INICIO DEL PROGRAMA PRINCIPAL---

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
        
        # Verificar si la imagen es realmente en escala de grises
        es_escala_grises = imagen_pil.mode in ['L', 'LA', '1', 'P']  # Modos de escala de grises en Pillow
        es_paleta_grises = False
        
        # Verificar adicionalmente para imágenes en modo 'P' (paleta)
        if imagen_pil.mode == 'P':
            # Obtener la paleta de colores
            paleta = imagen_pil.getpalette()
            if paleta:
                # Verificar si todos los colores en la paleta son tonos de gris
                es_paleta_grises = all(r == g == b for r, g, b in zip(paleta[::3], paleta[1::3], paleta[2::3]))
        
        if not (es_escala_grises or es_paleta_grises):
            st.error("❌ Error: Los filtros solo funcionan con imágenes en escala de grises.")
            
            # Ofrecer opción para convertir con advertencia
            if st.checkbox("Convertir mi imagen a escala de grises (⚠️CUIDADO⚠️)"):
                st.warning("""
                ⚠️ **Advertencia:** La imagen se convertirá a escala de grises. 
                Este proceso puede introducir ruido en la imagen debido a:
                - Pérdida de información cromática
                - Técnicas de conversión (promedio RGB, luminosidad, etc.)
                - Cuantización de valores
                """)
                
                # Convertir a modo 'L' (escala de grises estándar)
                imagen_pil = imagen_pil.convert("L")
                st.success("✅ ¡Conversión completada!")
                st.markdown("**Imagen convertida a escala de grises:**")
            else:
                st.info("ℹ️ Por favor, convierte tu imagen a escala de grises para continuar.")
                st.stop()
        else:
            # Si es escala de grises pero no en modo 'L', convertir a 'L'
            if imagen_pil.mode != 'L':
                st.info("ℹ️ La imagen está en un formato de escala de grises especial. Se convertirá al formato estándar.")
                imagen_pil = imagen_pil.convert("L")
            st.markdown("**Imagen cargada por el usuario:**")
    else:
        st.info("ℹ️ Por favor, sube una imagen para continuar.")
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


# CAMBIO CRUCIAL: Asegurarse de que la imagen_np sea float64 desde el inicio
# para que los filtros puedan producir valores fuera de [0, 255]
imagen_np = np.array(imagen_pil).astype(np.float64)

# Paso 4: Si es imagen de ejemplo y filtro de suavizado, aplicar ruido
if opcion_imagen == "Usar una imagen de ejemplo" and tipo_filtro == "Suavizado":
    # Seleccionar tipo de ruido
    tipo_ruido= "suave" 
    # Generar clave única para esta combinación imagen + tipo de ruido
    ruido_key = f"ruido_{imagen_seleccionada}_{tipo_ruido}"
    
    # Configurar parámetros de ruido
    if tipo_ruido == "extremo":
        intensidad = 0.12
        polarizacion = 40
    else:  # suave
        intensidad = 0.08
        polarizacion = 30
    
    if ruido_key not in st.session_state:
        # Asegurarse de pasar una copia para no modificar la original en sesión,
        # y convertir a uint8 para la función de ruido si es necesario
        imagen_ruido = agregar_ruido_sal_pimienta(
            imagen_np.copy().astype(np.uint8), 
            intensidad=intensidad,
            tipo=tipo_ruido,
            polarizacion=polarizacion
        )
        st.session_state[ruido_key] = imagen_ruido
        st.session_state["ultima_imagen"] = imagen_seleccionada
        st.session_state["ultimo_tipo_ruido"] = tipo_ruido
    
    imagen_np = st.session_state[ruido_key]
    st.image(imagen_np, caption=f"Imagen con ruido sal y pimienta", use_container_width=True)
else:
    st.image(imagen_pil, caption="Imagen original", use_container_width=True)

# Paso 5: Mostrar opciones de filtro
if tipo_filtro == "Suavizado":
    st.subheader("Comparación de Filtros de Suavizado")
    
    # Añadir CSS adicional
    st.markdown("""
    <style>
    .mask-container-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        margin: 0 auto;
    }
    .mask-table-suavizado {
        border-collapse: collapse;
        margin: 0 auto;
    }
    .mask-table-suavizado td {
        border: 1px solid #777;
        padding: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        width: 40px;
        height: 40px;
        background-color: #f0f2f6;
    }
    .fraction-text {
        font-size: 1.2em;
        margin: 5px 0;
        text-align: center;
    }
    .mask-title {
        text-align: center;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .action-buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 10px 0;
    }
    .action-buttons a {
        background-color: white;
        border: 1px solid black;
        color: black;
        padding: 6px 12px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    .action-buttons a:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown("<div class='mask-title'>Media Simple</div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<div class='mask-title'>Media Ponderada</div>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown("<div class='mask-title'>Mediana</div>", unsafe_allow_html=True)
        
        # Aplicar todos los filtros
        resultado_media_simple = filtro_media_simple(imagen_np, ksize=3)
        resultado_media_ponderada = filtro_media_ponderada(imagen_np)
        resultado_mediana = filtro_mediana(imagen_np, ksize=3)
        
        cols = st.columns([1, 1, 1])
        with cols[0]:
            # Convertir a uint8 para la visualización antes de mostrar
            st.image(resultado_media_simple.astype(np.uint8), use_container_width=True)
            st.markdown(
                f'<div class="action-buttons">'
                f'{get_image_download_link(resultado_media_simple.astype(np.uint8), "media_simple.png", "📥 Descargar")}'
                f'</div>', 
                unsafe_allow_html=True
            )
        with cols[1]:
            # Convertir a uint8 para la visualización antes de mostrar
            st.image(resultado_media_ponderada.astype(np.uint8), use_container_width=True)
            st.markdown(
                f'<div class="action-buttons">'
                f'{get_image_download_link(resultado_media_ponderada.astype(np.uint8), "media_ponderada.png", "📥 Descargar")}'
                f'</div>', 
                unsafe_allow_html=True
            )
        with cols[2]:
            # Convertir a uint8 para la visualización antes de mostrar
            st.image(resultado_mediana.astype(np.uint8), use_container_width=True)
            st.markdown(
                f'<div class="action-buttons">'
                f'{get_image_download_link(resultado_mediana.astype(np.uint8), "mediana.png", "📥 Descargar")}'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown("<div class='mask-container-center'>", unsafe_allow_html=True)
            st.markdown("<div class='fraction-text'>1/9</div>", unsafe_allow_html=True)
            mask_simple_values = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            table_html_simple = "<table class='mask-table-suavizado'>"
            for row in mask_simple_values:
                table_html_simple += "<tr>"
                for val in row:
                    table_html_simple += f"<td>{val}</td>"
                table_html_simple += "</tr>"
            table_html_simple += "</table>"
            st.markdown(table_html_simple, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<div class='mask-container-center'>", unsafe_allow_html=True)
            st.markdown("<div class='fraction-text'>1/16</div>", unsafe_allow_html=True)
            mask_ponderada_values = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            table_html_ponderada = "<table class='mask-table-suavizado'>"
            for row in mask_ponderada_values:
                table_html_ponderada += "<tr>"
                for val in row:
                    table_html_ponderada += f"<td>{val}</td>"
                table_html_ponderada += "</tr>"
            table_html_ponderada += "</table>"
            st.markdown(table_html_ponderada, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown("<div class='mask-container-center'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; margin: 10px 0;'>Filtro de la mediana</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; font-size: 0.9em;'>Reemplaza el valor central por la mediana de los vecinos</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Análisis comparativo
    with st.expander("🔍 Análisis Comparativo"):
        st.markdown("""
        **Comparación de resultados:**
        - ⚖️ **Media Simple:** Efectivo contra ruido aleatorio pero borra detalles
        - 🟦 **Media Ponderada:** Preserva mejor bordes que la media simple
        - 🎯 **Mediana:** Excelente contra ruido sal y pimienta, preserva bordes agudos
        """)


else: # Agudizado
    st.subheader("Selecciona el tipo de filtro de agudizado")
    tipo_agudizado = st.radio("Tipo de filtro:", ["Laplaciano", "Sobel"], horizontal=True)
    
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
        border: 1px solid #777; /* Borde, igual que suavizado */
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
    .action-buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 15px 0;
    }
    .action-buttons a {
        background-color: white;
        border: 1px solid black;
        color: black;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    .action-buttons a:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if tipo_agudizado == "Laplaciano":
        st.markdown("**Máscaras Laplacianas disponibles:**")
        
        col1, col2, col3 = st.columns(3)
        
        # Datos de las máscaras Laplacianas
        laplacian_masks = [
            {"title": "Clásico (4-vecinos)", "mask": [[0, 1, 0], [1, -4, 1], [0, 1, 0]], "key": "lap_clasico", "name": "Laplaciano clásico"},
            {"title": "8-positivo", "mask": [[1, 1, 1], [1, -8, 1], [1, 1, 1]], "key": "lap_positivo", "name": "Laplaciano 8-conectado positivo"},
            {"title": "8-negativo", "mask": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], "key": "lap_negativo", "name": "Laplaciano 8-conectado negativo"}
        ]

        for col, mask_info in zip([col1, col2, col3], laplacian_masks):
            with col:
                st.markdown(f"**{mask_info['title']}**")
                st.markdown("<div class='mask-container'>", unsafe_allow_html=True) 

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

                st.markdown("</div>", unsafe_allow_html=True) 


    else:  # Sobel
        st.markdown("**Máscaras Sobel disponibles:**")
        
        col1, col2 = st.columns(2)

        # Datos de las máscaras Sobel
        sobel_masks = [
            {"title": "Sobel X (horizontal)", "mask": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "key": "sobel_x", "name": "Sobel X"},
            {"title": "Sobel Y (vertical)", "mask": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "key": "sobel_y", "name": "Sobel Y"}
        ]

        for col, mask_info in zip([col1, col2], sobel_masks):
            with col:
                st.markdown(f"**{mask_info['title']}**")
                st.markdown("<div class='mask-container'>", unsafe_allow_html=True) 

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

                st.markdown("</div>", unsafe_allow_html=True)

    # Aplicar el filtro seleccionado
    if st.session_state.filtro_seleccionado:
        st.markdown(f"**Filtro seleccionado:** {st.session_state.filtro_seleccionado}")
        
        # El resultado 'raw' del filtro será ahora float64 con el rango completo de valores
        if "Laplaciano" in st.session_state.filtro_seleccionado:
            if "clásico" in st.session_state.filtro_seleccionado:
                resultado_raw = filtro_laplaciano(imagen_np, tipo='clasico')
            elif "positivo" in st.session_state.filtro_seleccionado:
                resultado_raw = filtro_laplaciano(imagen_np, tipo='8-conectado-positivo')
            else:
                resultado_raw = filtro_laplaciano(imagen_np, tipo='8-conectado-negativo')
        else: # Sobel
            if "X" in st.session_state.filtro_seleccionado:
                resultado_raw = filtro_sobel(imagen_np, direccion='x')
                # Para Sobel, la visualización de bordes a menudo usa la magnitud
                # Aunque el histograma 'sin normalizar' usará el resultado_raw (con negativos)
                # la imagen visualizada aquí será la magnitud normalizada
                resultado_para_normalizar = np.absolute(resultado_raw) 
            else:
                resultado_raw = filtro_sobel(imagen_np, direccion='y')
                resultado_para_normalizar = np.absolute(resultado_raw)
        
        # Normalizar para la VISUALIZACIÓN en pantalla
        # Si es Laplaciano, normalizamos el raw para visualización.
        # Si es Sobel, normalizamos la magnitud para visualización.
        if "Laplaciano" in st.session_state.filtro_seleccionado:
            resultado_normalizado = normalizar(resultado_raw)
        else: # Sobel
            resultado_normalizado = normalizar(resultado_para_normalizar)


        st.image(resultado_normalizado, caption="Bordes detectados (Normalizado para visualización)", use_container_width=True)
        
        # Añadir botones de acción para agudizado
        st.markdown(
            f'<div class="action-buttons">'
            f'{get_image_download_link(resultado_normalizado, "agudizado.png", "📥 Descargar imagen")}'
            f'</div>', 
            unsafe_allow_html=True
        )
        # En la sección de agudizado, después de mostrar la imagen filtrada y los botones de descarga:

        # ================ NUEVA SECCIÓN: ANÁLISIS DEL RE-ESCALAMIENTO ================
        st.subheader("🔬 Análisis Detallado del re-escalamiento")

        # Obtener los datos y estadísticas para el histograma sin normalizar usando la nueva función
        # Usamos resultado_raw que contiene los valores negativos y mayores a 255
        vals_sin_normalizar, min_val, max_val, mean_val, std_val = \
            get_image_data_for_histogram(resultado_raw)

        # Crear figura con tres subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Configurar estilo
        plt.style.use('ggplot')

        # Histograma 1: Imagen original
        ax1 = axes[0]
        vals_orig = imagen_np.ravel() # imagen_np es float64
        ax1.hist(vals_orig, bins=256, range=(0, 256), color='#3498db', alpha=0.8)
        ax1.set_title('Imagen Original', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Valor de Pixel', fontsize=10)
        ax1.set_ylabel('Frecuencia', fontsize=10)
        ax1.set_xlim([0, 255])
        ax1.set_yscale('log')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Añadir estadísticas al gráfico
        ax1.text(0.98, 0.95, 
                f"Mín: {np.min(imagen_np):.1f}\nMáx: {np.max(imagen_np):.1f}\nMedia: {np.mean(imagen_np):.1f}\nDesv: {np.std(imagen_np):.1f}",
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        # Histograma 2: Imagen filtrada sin normalizar (VALORES REALES DEL FILTRO)
        ax2 = axes[1]
        
        # Calcular los límites del eje X con un pequeño padding
        range_val = max_val - min_val
        padding = max(10.0, range_val * 0.05) 
        x_min_lim = min_val - padding
        x_max_lim = max_val + padding

        # Mostrar las estadísticas directamente en Streamlit (ahora deberían mostrar valores reales)
        st.markdown(f"#### Estadísticas de Imagen Filtrada (sin normalizar)")
        st.write(f"**Valor Mínimo:** {min_val:.2f}")
        st.write(f"**Valor Máximo:** {max_val:.2f}")
        if min_val < 0 or max_val > 255:
            st.warning("⚠️ **¡Atención!** Los valores de los píxeles están fuera del rango estándar [0, 255].")


        # Create bins that span the entire range with sufficient density
        num_bins = 1000  # Increased number of bins for better detail
        bins_sin_normalizar = np.linspace(x_min_lim, x_max_lim, num_bins)

        # Usar los bins definidos explícitamente y sin el parámetro 'range'
        ax2.hist(vals_sin_normalizar, bins=bins_sin_normalizar, color='#e74c3c', alpha=0.8) 
        ax2.set_title('Filtrada (sin normalizar)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Valor Real de Pixel', fontsize=10)
        ax2.set_ylabel('Frecuencia (log)', fontsize=10)
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.3)

        # Añadir línea vertical en 0 para referencia (si es relevante)
        if min_val < 0 and max_val > 0:
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Force X-axis limits to display negative and positive values
        ax2.autoscale(enable=False, axis='x') # Disable autoscale for x-axis
        ax2.set_xlim([x_min_lim, x_max_lim])

        # Añadir estadísticas precisas en el gráfico (con los mismos valores que los de arriba)
        stats_text = f"Mín: {min_val:.2f}\nMáx: {max_val:.2f}\nDesv: {std_val:.2f}"
        if min_val < 0 or max_val > 255:
            stats_text += "\n⚠️ Fuera de [0,255]"

        ax2.text(0.98, 0.95, stats_text,
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        # Histograma 3: Imagen filtrada normalizada
        ax3 = axes[2]
        # Usamos la data del resultado normalizado, que ya debería estar en uint8
        vals_normalizado, _, _, _, _ = get_image_data_for_histogram(resultado_normalizado)

        # Definir una pequeña tolerancia para los extremos (por problemas de float)
        TOLERANCE = 1e-6 # Una tolerancia pequeña para comparar floats con 0 o 255

        # Calcular porcentaje de píxeles en los extremos con tolerancia
        pixeles_min_extremo = np.sum(vals_normalizado <= (0 + TOLERANCE))
        pixeles_max_extremo = np.sum(vals_normalizado >= (255 - TOLERANCE))
        total_pixeles = len(vals_normalizado)
        
        # Asegurarse de que no dividimos por cero
        if total_pixeles > 0:
            porcentaje_min = (pixeles_min_extremo / total_pixeles) * 100
            porcentaje_max = (pixeles_max_extremo / total_pixeles) * 100
        else:
            porcentaje_min = 0.0
            porcentaje_max = 0.0


        # Calcular porcentaje de píxeles en bordes fuertes (0-umbral y 255-umbral)
        # Ajustamos el umbral para ser más inclusivo con la naturaleza de los bordes normalizados
        umbral_borde = 5 # Umbral más pequeño para ser más preciso cerca de los extremos
        pixeles_borde_oscuro = np.sum(vals_normalizado <= umbral_borde)
        pixeles_borde_claro = np.sum(vals_normalizado >= (255 - umbral_borde))
        
        # Considerar que los píxeles que están exactamente en los extremos (0 y 255) ya son "bordes fuertes"
        # y sumarlos si no se incluyen ya en el rango del umbral
        
        # Mejor, simplemente sumar los que estan en los rangos.
        if total_pixeles > 0:
            porcentaje_bordes = ((pixeles_borde_oscuro + pixeles_borde_claro) / total_pixeles) * 100
        else:
            porcentaje_bordes = 0.0
        

        # Enfocar en los extremos (0 y 255) para mostrar el efecto de bordes
        ax3.hist(vals_normalizado, bins=256, range=(0, 256), color='#2ecc71', alpha=0.8)
        ax3.set_title('Filtrada (normalizada)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Valor de Pixel', fontsize=10)
        ax3.set_ylabel('Frecuencia (log)', fontsize=10)
        ax3.set_xlim([0, 255])
        ax3.set_yscale('log')
        ax3.grid(True, linestyle='--', alpha=0.3)

        # Resaltar los extremos
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=255, color='black', linestyle='--', alpha=0.5)
        
        # Añadir líneas para el umbral de bordes
        ax3.axvline(x=umbral_borde, color='blue', linestyle=':', alpha=0.3)
        ax3.axvline(x=(255 - umbral_borde), color='blue', linestyle=':', alpha=0.3)


        # Añadir estadísticas con enfoque en los extremos
        ax3.text(0.98, 0.95, 
                f"Mín: {np.min(vals_normalizado):.1f}\nMáx: {np.max(vals_normalizado):.1f}\nDesv: {np.std(vals_normalizado):.1f}\n"
                f"Píxeles en 0: {porcentaje_min:.1f}%\n"
                f"Píxeles en 255: {porcentaje_max:.1f}%\n"
                f"Píxeles en bordes: {porcentaje_bordes:.1f}%", # Indicar el umbral
                transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        # Ajustar espacio entre subplots
        plt.tight_layout()

        # Mostrar en Streamlit
        st.pyplot(fig)

        # Explicación del proceso
        with st.expander("📝 Interpretación de los histogramas"):
            st.markdown("""
            **Proceso de transformación completo:**
            
            1. **Imagen Original:** Distribución típica de una imagen con valores concentrados en el rango medio
            2. **Filtrada (sin normalizar):** Resultado directo del filtro:
            - Valores negativos indican transiciones de claro a oscuro (bordes).
            - Valores positivos indican transiciones de oscuro a claro (bordes).
            - Los valores cercanos a cero indican áreas uniformes o de bajo cambio.
            - El rango dinámico es mayor que el rango de [0,255] - ¡Esto es la clave para entender la normalización!
            3. **Filtrada (normalizada):** Imagen final reescalada:
            - Concentración en extremos (0 y 255).
            - Concentración en medios
            
            **Características de los filtros de agudizado:**
            - 🎯 **Valores fuera de rango:** Los filtros ahora producen y muestran valores fuera de [0,255] antes de normalizar, evidenciando por qué es necesario el ajuste para la visualización.
            
            **Estadísticas clave:**
            - **Mín/Máx sin normalizar:** Estos valores ahora reflejan el rango completo y real de la salida del filtro.
            - **Píxeles en bordes:** Porcentaje de píxeles que representan bordes fuertes.
            - **Desviación Estándar:** Mide el contraste general (mayor = más bordes definidos).
            """)