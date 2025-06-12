# 🖼️ Procesamiento de Imágenes con Streamlit y OpenCV

Una aplicación web interactiva para el procesamiento de imágenes, desarrollada con `Streamlit` y `OpenCV` como parte del curso de Matemática Computacional en la Universidad Peruana de Ciencias Aplicadas (UPC). Esta herramienta permite explorar y aplicar diversos filtros de suavizado y agudizado para comprender mejor sus efectos en las imágenes.

## 🚀 Descripción del Proyecto

Este proyecto es una aplicación web interactiva diseñada para demostrar y aplicar diferentes técnicas de filtrado digital de imágenes. Los usuarios pueden experimentar con filtros de suavizado para reducir ruido y filtros de agudizado para realzar detalles y detectar bordes.

## ✨ Características Principales

* **Filtros de Suavizado**:
    * **Media Simple**: Aplica una máscara de convolución 3x3 de unos, normalizada por 1/9, para un suavizado básico.
    * **Media Ponderada**: Utiliza una máscara Gaussiana 3x3, normalizada por 1/16, para un suavizado más natural que considera la importancia de los píxeles vecinos.
    * **Mediana**: Implementa un filtro de mediana con una ventana 3x3, eficaz para eliminar ruido "sal y pimienta" sin distorsionar demasiado los bordes.

* **Filtros de Agudizado**:
    * **Operadores Laplacianos**: Incluye 3 variantes de máscaras Laplacianas (clásica de 4 vecinos, 8-conectada positiva y 8-conectada negativa) para realzar bordes y detalles finos.
    * **Operadores Sobel**: Permite la detección de bordes en direcciones horizontal (Sobel X) y vertical (Sobel Y), fundamentales en el análisis de imágenes.

* **Gestión de Imágenes**:
    * **Imágenes de Ejemplo**: Incluye una colección de imágenes predefinidas para facilitar la demostración y prueba de los filtros.
    * **Carga de Imágenes Personalizadas**: Ofrece la opción de subir tus propias imágenes (formatos JPG, PNG, JPEG) para un análisis personalizado.

* **Visualización de Máscaras (Kernels)**:
    * Cada filtro convolucional (Media Simple, Media Ponderada, Laplaciano, Sobel) muestra una representación gráfica de su máscara (kernel) utilizada para el procesamiento, facilitando la comprensión de su operación.

* **Ruido Controlado**:
    * Para las imágenes de ejemplo en el modo de suavizado, la aplicación puede aplicar automáticamente ruido "sal y pimienta" de forma controlada, permitiendo evaluar la efectividad de los filtros de reducción de ruido.

## 🖥️ Uso de la Aplicación

Para utilizar la aplicación, sigue estos sencillos pasos:

1.  **Selección de Imagen**:
    * Al iniciar, decide si deseas trabajar con una de las **imágenes de ejemplo** provistas o **subir tu propia imagen**.
    * Todas las imágenes son convertidas automáticamente a escala de grises para su procesamiento.

2.  **Selección de Tipo de Filtro**:
    * Elige entre **"Suavizado"** (para reducir ruido y suavizar la imagen) o **"Agudizado"** (para destacar bordes y realzar detalles).

3.  **Aplicación y Visualización de Filtros**:
    * Una vez seleccionado el tipo de filtro, podrás elegir entre las diferentes variantes disponibles (por ejemplo, Media Simple, Media Ponderada, Mediana para suavizado; Laplaciano, Sobel para agudizado).
    * La aplicación mostrará la **máscara (kernel)** correspondiente al filtro seleccionado (donde aplique).
    * Observa cómo la **imagen procesada** se actualiza en tiempo real, permitiéndote comparar el resultado con la imagen original.

4.  **Ejecuta la aplicación Streamlit**:
    ```bash
    streamlit run app.py
    ```

    Esto abrirá la aplicación en tu navegador web predeterminado.
