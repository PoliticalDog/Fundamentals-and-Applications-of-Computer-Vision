from tkinter import simpledialog  # (usado por tu versión original)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import skew
from math import log2

# ======== Lógica (idéntica a tu código, sin UI Tk) ========

def cargar_imagen(ruta):
    imagen = Image.open(ruta)
    plt.imshow(imagen); plt.title("Imagen original"); plt.axis("off"); plt.show()
    return imagen

def separar_rgb(imagen):
    r, g, b = imagen.split()
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.imshow(r, cmap='Reds');   plt.title("Componente R"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(g, cmap='Greens'); plt.title("Componente G"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(b, cmap='Blues');  plt.title("Componente B"); plt.axis("off")
    plt.show()

def convertir_a_grises(imagen):
    imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    plt.imshow(imagen_cv, cmap='gray'); plt.title("Imagen en escala de grises"); plt.axis("off"); plt.show()
    return imagen_cv

def binarizar_imagen(imagen_gris, umbral=128):
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    plt.imshow(binaria, cmap='gray'); plt.title(f"Binarizada (umbral={umbral})"); plt.axis("off"); plt.show()

def convertir_a_cmyk(imagen_pil):
    cmyk = imagen_pil.convert("CMYK")
    c, m, y, k = cmyk.split()
    plt.figure(figsize=(12,4))
    for i, (chan, title) in enumerate(zip([c,m,y,k], ["Cyan","Magenta","Yellow","Black"]), 1):
        plt.subplot(1,4,i); plt.imshow(chan, cmap='gray'); plt.title(title); plt.axis("off")
    plt.show()
    return cmyk

def convertir_a_hsl(imagen_pil):
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(img_cv)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(h, cmap='hsv');  plt.title("Hue");       plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(l, cmap='gray'); plt.title("Lightness"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(s, cmap='gray'); plt.title("Saturation");plt.axis("off")
    plt.show()
    return img_cv

def binarizar_imagen_umbral(imagen_gris, umbral):
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    plt.imshow(binaria, cmap='gray'); plt.title(f"Binarizada (umbral={umbral})"); plt.axis("off"); plt.show()
    return binaria

def binarizar_imagen_otsu(imagen_gris):
    _, binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(binaria, cmap='gray'); plt.title("Binarizada (Otsu)"); plt.axis("off"); plt.show()
    return binaria

def calcular_histogramas_rgb(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = {}
    colores = {"Rojo":"red","Verde":"green","Azul":"blue"}

    for i, canal in enumerate(['Rojo','Verde','Azul']):
        datos = imagen_rgb[:,:,i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0,256))
        prob = histograma / histograma.sum()
        energia = np.sum(prob ** 2)
        entropia = -np.sum([p*log2(p) for p in prob if p>0])
        asimetria = skew(datos)
        media = np.mean(datos); varianza = np.var(datos)
        resultados[canal] = {'Energía': energia, 'Entropía': entropia, 'Asimetría': asimetria, 'Media': media, 'Varianza': varianza}
        plt.figure(); plt.title(f'Histograma {canal}'); plt.xlabel('Intensidad'); plt.ylabel('Frecuencia')
        plt.plot(histograma, color=colores[canal]); plt.grid(True); plt.show()

    for canal, props in resultados.items():
        print(f'\nCanal {canal}:')
        for prop, valor in props.items():
            print(f'  {prop}: {valor:.4f}')

def calcular_histograma_grises(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    datos = imagen.flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0,256))
    prob = histograma / histograma.sum()
    energia = np.sum(prob ** 2)
    entropia = -np.sum([p*log2(p) for p in prob if p>0])
    asimetria = skew(datos)
    media = np.mean(datos); varianza = np.var(datos)
    print("\nPropiedades (grises):")
    print(f" Energía: {energia:.4f}"); print(f" Entropía: {entropia:.4f}"); print(f" Asimetría: {asimetria:.4f}")
    print(f" Media: {media:.2f}"); print(f" Varianza: {varianza:.2f}")
    plt.figure(); plt.title('Histograma (grises)'); plt.xlabel('Intensidad'); plt.ylabel('Frecuencia')
    plt.plot(histograma, color='gray'); plt.grid(True); plt.show()

# Helpers para interfaz
def compute_histogramas_rgb_arrays(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None: return None
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    out = {}
    for i, canal in enumerate(['Rojo','Verde','Azul']):
        datos = rgb[:,:,i].flatten()
        h,_ = np.histogram(datos, bins=256, range=(0,256))
        out[canal] = h
    return out

def compute_histograma_gris_array(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    datos = img.flatten()
    h,_ = np.histogram(datos, bins=256, range=(0,256))
    return h
