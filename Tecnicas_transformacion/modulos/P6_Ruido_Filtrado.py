# modulos/P6_Ruido_Filtrado.py
import cv2
import numpy as np


# ================= Helpers =================

def to_gray(img_bgr):
    """
    Asegura imagen en escala de grises (uint8).
    """
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        gray = img_bgr
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)


# ================= Ruido =================

def add_salt_pepper(img, amount=0.02):
    """
    Agrega ruido sal y pimienta.
    amount: proporción de píxeles afectados (0–1).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None

    out = img.copy()
    h, w = out.shape[:2]
    num_pixels = int(amount * h * w)

    # coordenadas aleatorias
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)

    # mitad sal, mitad pimienta
    half = num_pixels // 2
    out[ys[:half], xs[:half]] = 0      # pimienta
    out[ys[half:], xs[half:]] = 255    # sal
    return out


def add_salt(img, amount=0.02):
    """
    Agrega solo ruido de sal (píxeles blancos).
    amount: proporción de píxeles afectados (0–1).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None

    out = img.copy()
    h, w = out.shape[:2]
    num_pixels = int(amount * h * w)

    # coordenadas aleatorias
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)

    out[ys, xs] = 255  # sal (blanco)
    return out


def add_pepper(img, amount=0.02):
    """
    Agrega solo ruido de pimienta (píxeles negros).
    amount: proporción de píxeles afectados (0–1).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None

    out = img.copy()
    h, w = out.shape[:2]
    num_pixels = int(amount * h * w)

    # coordenadas aleatorias
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)

    out[ys, xs] = 0  # pimienta (negro)
    return out


def add_gaussian_noise(img, mean=0, sigma=20):
    """
    Agrega ruido gaussiano aditivo.
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None

    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# ================= Filtros lineales =================

def mean_filter(img, ksize=5):
    """
    Filtro promediador (blur).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None
    return cv2.blur(img, (ksize, ksize))


def weighted_mean_filter(img, ksize=5):
    """
    Filtro promediador pesado (kernel centrado).
    Si ksize != 3, se usa blur normal para simplificar.
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None
    if ksize != 3:
        return cv2.blur(img, (ksize, ksize))

    # kernel 3x3 con más peso en el centro
    kernel = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=np.float32
    )
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def gaussian_filter(img, ksize=5, sigma=0):
    """
    Filtro gaussiano.
    ksize debe ser impar.
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """
    Filtro bilateral (suaviza preservando bordes).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ================= Filtros no lineales =================

def median_filter(img, ksize=5):
    """
    Filtro de mediana (muy bueno para sal y pimienta).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def mode_filter(img, ksize=3):
    """
    Filtro de moda en una vecindad ksize x ksize.
    Implementado “a mano” con ventana deslizante.
    (No es súper eficiente pero sirve para la práctica.)
    Funciona con imágenes en grises. Para imágenes a color, se convierte a grises.
    """
    if img is None:
        return None
    
    # El filtro de moda funciona mejor en escala de grises
    gray = to_gray(img)
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = gray.shape
    out = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            window = padded[y:y+ksize, x:x+ksize]
            vals, counts = np.unique(window, return_counts=True)
            out[y, x] = vals[np.argmax(counts)]
    return out


def max_filter(img, ksize=3):
    """
    Filtro de máximo (elimina puntos oscuros).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel)


def min_filter(img, ksize=3):
    """
    Filtro de mínimo (elimina puntos brillantes).
    Funciona tanto con imágenes en grises como a color (BGR).
    """
    if img is None:
        return None
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel)
