#  IMAGE ANALYSIS  
## Prácticas de Visión por Computadora y Procesamiento Digital de Imágenes

Este repositorio contiene una **colección estructurada de prácticas en Python** orientadas al estudio de técnicas fundamentales de **Procesamiento Digital de Imágenes (PDI)** y **Visión por Computadora**, desarrolladas con fines **académicos y experimentales**.

El proyecto integra procesamiento en el **dominio espacial** y en el **dominio de la frecuencia**, abarcando desde histogramas y binarización hasta técnicas avanzadas como **FFT** y **DCT**, todo organizado en módulos reutilizables y una **interfaz gráfica unificada**.

---

##  Objetivos del proyecto

- Comprender y aplicar técnicas clásicas de PDI.
- Analizar el efecto del ruido y los filtros espaciales.
- Explorar el dominio de la frecuencia para realce y filtrado.
- Implementar una arquitectura modular y escalable en Python.
- Centralizar las prácticas mediante una interfaz gráfica común.

---

##  Prácticas implementadas

- **Histogramas e intensidades**
  - Histogramas RGB
  - Binarización
- **Operaciones lógicas y relacionales**
- **Etiquetado de regiones**
  - Conectividad 4 y 8
- **Pseudocolor**
  - Mapas de color para realce visual
- **Morfología matemática**
  - Erosión, dilatación, apertura y cierre
- **Ruido y filtrado**
  - Ruido sal y pimienta
  - Ruido gaussiano
  - Filtros espaciales lineales y no lineales
- **Dominio de la frecuencia**
  - Transformada Rápida de Fourier (FFT)
  - Filtrado frecuencial (ideal, gaussiano, Butterworth)
  - Transformada Discreta del Coseno (DCT)
  - Reconstrucción de imágenes

---

##  Estructura del proyecto

```text
IMAGE ANALYSIS/
│
├── garbage-dataset/              # Dataset externo (ignorado por git)
├── Imagenes propias/             # Imágenes de prueba
├── Imagenes Resultados/          # Resultados generados
├── Tecnicas transformacion/      # Salidas intermedias
│
├── interfaz/
│   ├── __init__.py
│   └── interfaz.py               # Interfaz gráfica principal
│
├── modulos/
│   ├── __init__.py
│   ├── P1_Histograma_RGB_BIN.py
│   ├── P2_Logicas_Relacionales.py
│   ├── P3_Etiquetado_4_8.py
│   ├── P4_Pseudocolor.py
│   ├── P5_Morfologia.py
│   ├── P6_Ruido_Filtrado.py
│   ├── P7_Frecuencia.py
│   └── P7_practica_frecuencia_ISC_extension_prof.py
│
├── main.py                       # Punto de entrada del proyecto
├── .gitignore
└── README.md


---

## Requisitos

- Python 3.8 o superior
- NumPy
- OpenCV (opencv-python)
- Matplotlib
- SciPy
- Pillow

Instalación rápida:
pip install numpy opencv-python matplotlib scipy pillow

---

 ## NOTAS:
Las carpetas de resultados e imágenes de prueba se generan dinámicamente.

El proyecto está diseñado para experimentación académica, no para producción.

La estructura modular facilita la extensión con nuevas prácticas de PDI.

---