# IMAGE ANALYSIS  
## Prácticas de Visión por Computadora y Procesamiento Digital de Imágenes

Este repositorio contiene una colección de prácticas desarrolladas en **Python** para el estudio de técnicas fundamentales de **Análisis y Procesamiento Digital de Imágenes**, realizadas como parte de una materia académica de Visión por Computadora.

El proyecto integra métodos de procesamiento espacial y en el dominio de la frecuencia, abarcando desde histogramas y binarización hasta transformadas espectrales como **FFT** y **DCT**.

---

---

##  Prácticas implementadas

- Análisis de histogramas e intensidades
- Binarización de imágenes
- Operaciones lógicas y relacionales
- Etiquetado de regiones (4 y 8 conectividades)
- Pseudocolor y mapas de color
- Morfología matemática
- Ruido en imágenes y filtrado espacial
- Procesamiento en el dominio de la frecuencia:
  - FFT y filtrado frecuencial
  - DCT y reconstrucción de imagen

---

##  Requisitos

- Python 3.8 o superior
- NumPy
- OpenCV
- Matplotlib
- SciPy

---

##  Ejecución

Se recomienda ejecutar desde P7_Frecuencia/main.py, ya que contiene todas las prácticas unificadas en ese módulo corriendo la interfaz general.
---


##  Estructura del proyecto

IMAGE ANALYSIS/
│
├── P1_RGB_BIN_MODELOS_HISTOGRAMAS/
│ Análisis de histogramas RGB, binarización y modelos de intensidad
│
├── P2yP3_Etiquetados_Operaciones/
│ Operaciones lógicas, relacionales y etiquetado de regiones (4 y 8 conectividades)
│
├── P4_MapaColores/
│ Técnicas de pseudocolor y visualización mediante mapas de color
│
├── P5_Morfologia/
│ Morfología matemática: erosión, dilatación, apertura y cierre
│
├── P6_Ruido_Filtrado/
│ Modelado de ruido (sal y pimienta, gaussiano) y filtrado espacial
│
├── P7_Frecuencia/
│ Análisis en el dominio de la frecuencia:
│ - Transformada Rápida de Fourier (FFT)
│ - Filtrado en frecuencia (ideal, gaussiano, Butterworth)
│ - Transformada Discreta del Coseno (DCT)
│
├── Imagenes propias/
│ Imágenes de prueba utilizadas en las prácticas
│
├── Imagenes Resultados/
│ Resultados generados por los algoritmos
│
├── garbage-dataset/ (ignorado por git)
│ Dataset externo de gran tamaño  para jugar con metodos(~20k imágenes) --> https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?select=garbage-dataset
│
├── .gitignore
└── README.md
