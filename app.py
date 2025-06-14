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
from matplotlib.ticker import LogFormatter # Importar LogFormatter

plt.rcParams['text.usetex'] = False

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

# Función para cargar y procesar imágenes de ejemplo
@st.cache_data
def load_example_image(folder, image_name):
    path = os.path.join("imagenes_ejemplo", folder, image_name)
    img_pil = Image.open(path).convert("L") # Asegurar escala de grises
    return np.array(img_pil) # Devolver como uint8 para empezar

# Paso 3: Cargar imagen según la opción elegida
imagen_original_np = None 
if opcion_imagen == "Subir mi propia imagen":
    uploaded_file = st.file_uploader("Sube tu propia imagen (JPG o PNG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        imagen_pil = Image.open(uploaded_file)
        # Verificar y convertir a escala de grises si no lo está
        es_escala_grises = imagen_pil.mode in ['L', 'LA', '1', 'P']
        es_paleta_grises = False
        if imagen_pil.mode == 'P':
            paleta = imagen_pil.getpalette()
            if paleta:
                es_paleta_grises = all(r == g == b for r, g, b in zip(paleta[::3], paleta[1::3], paleta[2::3]))
        
        if not (es_escala_grises or es_paleta_grises):
            st.error("❌ Error: Los filtros solo funcionan con imágenes en escala de grises.")
            if st.checkbox("Convertir mi imagen a escala de grises (⚠️CUIDADO⚠️)"):
                st.warning("""
                ⚠️ **Advertencia:** La imagen se convertirá a escala de grises. 
                Este proceso puede introducir ruido en la imagen debido a:
                - Pérdida de información cromática
                - Técnicas de conversión (promedio RGB, luminosidad, etc.)
                - Cuantización de valores
                """)
                imagen_pil = imagen_pil.convert("L")
                st.success("✅ ¡Conversión completada!")
                st.markdown("**Imagen convertida a escala de grises:**")
            else:
                st.info("ℹ️ Por favor, convierte tu imagen a escala de grises para continuar.")
                st.stop()
        else:
            if imagen_pil.mode != 'L':
                st.info("ℹ️ La imagen está en un formato de escala de grises especial. Se convertirá al formato estándar.")
                imagen_pil = imagen_pil.convert("L")
            st.markdown("**Imagen cargada por el usuario:**")
        imagen_original_np = np.array(imagen_pil).astype(np.uint8) # Imagen base siempre uint8
        st.image(imagen_original_np, caption="Imagen original", use_container_width=True)
    else:
        st.info("ℹ️ Por favor, sube una imagen para continuar.")
        st.stop()
else: # Usar una imagen de ejemplo
    folder = "suavizado" if tipo_filtro == "Suavizado" else "agudizado"
    imagenes_ejemplo = [f for f in os.listdir(os.path.join("imagenes_ejemplo", folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    opciones_sin_extension = [os.path.splitext(f)[0] for f in imagenes_ejemplo]
    opcion_seleccionada = st.selectbox("Elige una imagen de ejemplo", opciones_sin_extension)
    imagen_seleccionada_completa = next((f for f in imagenes_ejemplo if os.path.splitext(f)[0] == opcion_seleccionada), None)
    if imagen_seleccionada_completa:
        imagen_original_np = load_example_image(folder, imagen_seleccionada_completa) # Cargada y cacheada como uint8
        st.markdown("**Imagen de ejemplo seleccionada:**")
    else:
        st.error("No se encontró la imagen seleccionada")
        st.stop()

imagen_para_procesar = imagen_original_np.copy()

# Paso 4: Si es imagen de ejemplo y filtro de suavizado, aplicar ruido
if opcion_imagen == "Usar una imagen de ejemplo" and tipo_filtro == "Suavizado":
    tipo_ruido= "suave" 
    ruido_key = f"ruido_{imagen_seleccionada_completa}_{tipo_ruido}" # Usar nombre completo para clave
    
    if tipo_ruido == "extremo":
        intensidad = 0.12
        polarizacion = 40
    else:  # suave
        intensidad = 0.08
        polarizacion = 30
    
    if ruido_key not in st.session_state:
        with st.spinner("Aplicando ruido a la imagen..."):
            imagen_con_ruido = agregar_ruido_sal_pimienta(
                imagen_original_np, # pasar la original uint8
                intensidad=intensidad,
                tipo=tipo_ruido,
                polarizacion=polarizacion
            )
        st.session_state[ruido_key] = imagen_con_ruido
        st.session_state["ultima_imagen_ruido_aplicada"] = imagen_seleccionada_completa # Almacenar para control
        st.session_state["ultimo_tipo_ruido_aplicado"] = tipo_ruido
    
    imagen_para_procesar = st.session_state[ruido_key]
    st.image(imagen_para_procesar, caption=f"Imagen con ruido sal y pimienta", use_container_width=True)
else:
    pass


# Paso 5: Mostrar opciones de filtro y aplicar
if tipo_filtro == "Suavizado":
    st.subheader("Comparación de Filtros de Suavizado")
    
    st.markdown("""
    <style>
    .mask-container-center { display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; margin: 0 auto; }
    .mask-table-suavizado { border-collapse: collapse; margin: 0 auto; }
    .mask-table-suavizado td { border: 1px solid #777; padding: 8px; text-align: center; font-weight: bold; font-size: 16px; width: 40px; height: 40px; background-color: #f0f2f6; }
    .fraction-text { font-size: 1.2em; margin: 5px 0; text-align: center; }
    .mask-title { text-align: center; font-weight: bold; margin-bottom: 5px; }
    .action-buttons { display: flex; justify-content: center; gap: 10px; margin: 10px 0; }
    .action-buttons a { background-color: white; border: 1px solid black; color: black; padding: 6px 12px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; border-radius: 4px; transition: background-color 0.3s; }
    .action-buttons a:hover { background-color: #45a049; }
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
        with st.spinner("Aplicando filtros de suavizado..."):
            resultado_media_simple = filtro_media_simple(imagen_para_procesar, ksize=3)
            resultado_media_ponderada = filtro_media_ponderada(imagen_para_procesar)
            resultado_mediana = filtro_mediana(imagen_para_procesar, ksize=3)
        
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.image(resultado_media_simple, use_container_width=True)
            st.markdown(
                f'<div class="action-buttons">'
                f'{get_image_download_link(resultado_media_simple, "media_simple.png", "📥 Descargar")}'
                f'</div>', 
                unsafe_allow_html=True
            )
        with cols[1]:
            st.image(resultado_media_ponderada, use_container_width=True)
            st.markdown(
                f'<div class="action-buttons">'
                f'{get_image_download_link(resultado_media_ponderada, "media_ponderada.png", "📥 Descargar")}'
                f'</div>', 
                unsafe_allow_html=True
            )
        with cols[2]:
            st.image(resultado_mediana, use_container_width=True)
            st.markdown(
                f'<div class="action-buttons">'
                f'{get_image_download_link(resultado_mediana, "mediana.png", "📥 Descargar")}'
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
    
    st.markdown("""
    <style>
    .mask-table { left: 0; border-collapse: collapse; width: 120px; }
    .mask-table td { border: 1px solid #777; padding: 8px; text-align: center; font-weight: bold; font-size: 16px; width: 40px; height: 40px; background-color: #f0f2f6; }
    .mask-container { display: flex; flex-direction: column; align-items: center; margin-bottom: 15px; }
    .action-buttons { display: flex; justify-content: center; gap: 10px; margin: 15px 0; }
    .action-buttons a { background-color: white; border: 1px solid black; color: black; padding: 8px 16px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; border-radius: 4px; transition: background-color 0.3s; }
    .action-buttons a:hover { background-color: #45a049; }
    </style>
    """, unsafe_allow_html=True)
    
    if tipo_agudizado == "Laplaciano":
        st.markdown("**Máscaras Laplacianas disponibles:**")
        
        col1, col2, col3 = st.columns(3)
        
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
        
        imagen_para_filtro_agudizado = imagen_original_np.astype(np.float64)

        with st.spinner("Aplicando filtro de agudizado..."):
            if "Laplaciano" in st.session_state.filtro_seleccionado:
                if "clásico" in st.session_state.filtro_seleccionado:
                    resultado_raw = filtro_laplaciano(imagen_para_filtro_agudizado, tipo='clasico')
                elif "positivo" in st.session_state.filtro_seleccionado:
                    resultado_raw = filtro_laplaciano(imagen_para_filtro_agudizado, tipo='8-conectado-positivo')
                else:
                    resultado_raw = filtro_laplaciano(imagen_para_filtro_agudizado, tipo='8-conectado-negativo')
            else: # Sobel
                if "X" in st.session_state.filtro_seleccionado:
                    resultado_raw = filtro_sobel(imagen_para_filtro_agudizado, direccion='x')
                    resultado_para_normalizar = np.absolute(resultado_raw) 
                else:
                    resultado_raw = filtro_sobel(imagen_para_filtro_agudizado, direccion='y')
                    resultado_para_normalizar = np.absolute(resultado_raw)
        
        # Normalizar para la VISUALIZACIÓN en pantalla
        if "Laplaciano" in st.session_state.filtro_seleccionado:
            resultado_normalizado = normalizar(resultado_raw)
        else: # Sobel
            resultado_normalizado = normalizar(resultado_para_normalizar)


        st.image(resultado_normalizado, caption="Bordes detectados (Normalizado para visualización)", use_container_width=True)
        
        st.markdown(
            f'<div class="action-buttons">'
            f'{get_image_download_link(resultado_normalizado, "agudizado.png", "📥 Descargar imagen")}'
            f'</div>', 
            unsafe_allow_html=True
        )

        #Histogramas
        st.subheader("🔬 Análisis Detallado del Re-escalamiento")

        vals_sin_normalizar, min_val, max_val, mean_val, std_val = \
            get_image_data_for_histogram(resultado_raw)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        plt.style.use('ggplot')

        ax1 = axes[0]
        vals_orig = imagen_original_np.ravel()
        ax1.hist(vals_orig, bins=256, range=(0, 256), color='#3498db', alpha=0.8)
        ax1.set_title('Imagen Original', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Valor de Pixel', fontsize=10)
        ax1.set_ylabel('Frecuencia', fontsize=10)
        ax1.set_xlim([0, 255])
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False)) # Aplicar formateador de log
        ax1.grid(True, linestyle='--', alpha=0.3)

        ax1.text(0.98, 0.95, 
                f"Mín: {np.min(imagen_original_np):.1f}\nMáx: {np.max(imagen_original_np):.1f}\nDesv: {np.std(imagen_original_np):.1f}",
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        ax2 = axes[1]
        
        range_val = max_val - min_val
        padding = max(10.0, range_val * 0.05) 
        x_min_lim = min_val - padding
        x_max_lim = max_val + padding

        st.markdown(f"#### Estadísticas de Imagen Filtrada (sin normalizar)")
        st.write(f"**Valor Mínimo:** {min_val:.2f}")
        st.write(f"**Valor Máximo:** {max_val:.2f}")
        st.write(f"**Desviación Estándar:** {std_val:.2f}")
        if min_val < 0 or max_val > 255:
            st.warning("⚠️ **¡Atención!** Los valores de los píxeles están fuera del rango estándar [0, 255].")
            st.info("ℹ️ **Nota:** Estos son los valores *reales* de salida del filtro antes de la normalización. Muestran por qué es necesaria la normalización para visualizar la imagen correctamente.")


        num_bins = 1000 
        bins_sin_normalizar = np.linspace(x_min_lim, x_max_lim, num_bins)

        ax2.hist(vals_sin_normalizar, bins=bins_sin_normalizar, color='#e74c3c', alpha=0.8) 
        ax2.set_title('Filtrada (sin normalizar)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Valor Real de Pixel', fontsize=10)
        ax2.set_ylabel('Frecuencia (log)', fontsize=10)
        ax2.set_yscale('log')
        ax2.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False)) # Aplicar formateador de log
        ax2.grid(True, linestyle='--', alpha=0.3)

        if min_val < 0 and max_val > 0:
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        ax2.autoscale(enable=False, axis='x') 
        ax2.set_xlim([x_min_lim, x_max_lim])

        stats_text = f"Mín: {min_val:.2f}\nMáx: {max_val:.2f}\nDesv: {std_val:.2f}"
        if min_val < 0 or max_val > 255:
            stats_text += "\n⚠️ Fuera de [0,255]"

        ax2.text(0.98, 0.95, stats_text,
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        ax3 = axes[2]
        vals_normalizado, _, _, _, _ = get_image_data_for_histogram(resultado_normalizado)

        TOLERANCE = 1e-6 

        pixeles_min_extremo = np.sum(vals_normalizado <= (0 + TOLERANCE))
        pixeles_max_extremo = np.sum(vals_normalizado >= (255 - TOLERANCE))
        total_pixeles = len(vals_normalizado)
        
        if total_pixeles > 0:
            porcentaje_min = (pixeles_min_extremo / total_pixeles) * 100
            porcentaje_max = (pixeles_max_extremo / total_pixeles) * 100
        else:
            porcentaje_min = 0.0
            porcentaje_max = 0.0

        umbral_borde = 5 
        pixeles_borde_oscuro = np.sum(vals_normalizado <= umbral_borde)
        pixeles_borde_claro = np.sum(vals_normalizado >= (255 - umbral_borde))
        
        if total_pixeles > 0:
            porcentaje_bordes = ((pixeles_borde_oscuro + pixeles_borde_claro) / total_pixeles) * 100
        else:
            porcentaje_bordes = 0.0
        

        ax3.hist(vals_normalizado, bins=256, range=(0, 256), color='#2ecc71', alpha=0.8)
        ax3.set_title('Filtrada (normalizada)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Valor de Pixel', fontsize=10)
        ax3.set_ylabel('Frecuencia (log)', fontsize=10)
        ax3.set_xlim([0, 255])
        ax3.set_yscale('log')
        ax3.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False)) # Aplicar formateador de log
        ax3.grid(True, linestyle='--', alpha=0.3)

        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=255, color='black', linestyle='--', alpha=0.5)
        
        ax3.axvline(x=umbral_borde, color='blue', linestyle=':', alpha=0.3)
        ax3.axvline(x=(255 - umbral_borde), color='blue', linestyle=':', alpha=0.3)


        stats_text = f"Mín: {np.min(vals_normalizado):.1f}\nMáx: {np.max(vals_normalizado):.1f}\n" \
                     f"Píxeles en 0: {porcentaje_min:.1f}%\n" \
                     f"Píxeles en 255: {porcentaje_max:.1f}%\n" \
                     f"Píxeles en bordes: {porcentaje_bordes:.1f}%"
        ax3.text(0.98, 0.95, 
                stats_text,
                transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        plt.tight_layout()

        st.pyplot(fig)

        with st.expander("📝 Interpretación de los histogramas"):
            st.markdown("""
            **Proceso de transformación completo:**
            
            1. **Imagen Original:** Distribución típica de una imagen con valores concentrados en el rango medio
            2. **Filtrada (sin normalizar):** Resultado directo del filtro.
            - Valores negativos indican transiciones de claro a oscuro (bordes).
            - Valores positivos indican transiciones de oscuro a claro (bordes).
            - Los valores cercanos a cero indican áreas uniformes o de bajo cambio.
            - El rango dinámico es mayor que [0,255] - ¡Esto es la clave para entender la normalización!
            3. **Filtrada (normalizada):** Imagen final reescalada:
            - Concentración en extremos (0 y 255).
            - Concentración en medios.
            
            **Características de los filtros de agudizado:**
            - 🎯 **Valores fuera de rango:** Los filtros ahora producen y muestran valores fuera de [0,255] antes de normalizar, evidenciando por qué es necesario el ajuste para la visualización.
            - 📊 **Efecto de bordes:** Después de normalizar, los bordes se concentran en los extremos 0 (bordes oscuros) y 255 (bordes claros), haciendo la imagen visible.
            
            **Estadísticas clave:**
            - **Mín/Máx sin normalizar:** Estos valores ahora reflejan el rango completo y real de la salida del filtro.
            - **Píxeles en bordes:** Porcentaje de píxeles que representan bordes fuertes. Ahora se calcula con una tolerancia para mayor precisión.
            - **Desviación Estándar:** Mide el contraste general (mayor = más bordes definidos).
            """)
