# modulos/P8_Frecuencia.py
# Wrapper para usar el código de la profesora (practica_frecuencia_ISC.py)

import os

from practica_frecuencia_ISC_extension_profesora import (
    cargar_imagen,
    mostrar_fft,
    mostrar_dct,
)



def ejecutar_desde_gui(
    ruta_imagen=None,
    filtro="butterworth",
    tipo="lowpass",
    cutoff=0.15,
    orden=2,
    dct_q=0.5,
    carpeta_salida="salidas_frecuencia_gui",
):
    """
    Ejecuta la práctica de frecuencia (FFT + DCT) usando el código de la profesora,
    pero pensada para llamarse desde la interfaz gráfica.

    - ruta_imagen: path a la imagen (o None para usar la sintética de prueba).
    - filtro: 'ideal', 'gaussiano', 'butterworth'
    - tipo: 'lowpass' o 'highpass'
    - cutoff: float, radio de corte normalizado (0–0.5 aprox).
    - orden: entero (solo para Butterworth).
    - dct_q: factor de cuantización para DCT (0.3–1.0 típico).
    - carpeta_salida: carpeta donde se guardan las figuras generadas.

    Devuelve: ruta de la carpeta de salida.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    # Usamos exactamente la función de la profa
    img = cargar_imagen(ruta_imagen)

    # FFT + filtrado
    path_fft = os.path.join(carpeta_salida, "fft_filtrado_gui.png")
    mostrar_fft(
        img,
        filtro=filtro,
        tipo=tipo,
        cutoff=cutoff,
        orden=orden,
        guardar=path_fft,
    )

    # DCT + reconstrucción
    path_dct = os.path.join(carpeta_salida, "dct_reconstruccion_gui.png")
    mostrar_dct(
        img,
        q_factor=dct_q,
        guardar=path_dct,
    )

    return carpeta_salida, path_fft, path_dct
