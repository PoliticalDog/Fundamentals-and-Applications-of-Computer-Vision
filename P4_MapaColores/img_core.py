# img_core.py
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import skew
from math import log2

# ========= Conversión / Mosaicos de visualización =========
def to_gray_if_needed(img_bgr):
    """Convierte a gris si es BGR, si ya es gris regresa igual."""
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

def mosaic_rgb_channels(img_bgr):
    """
    Devuelve una imagen BGR con un mosaico horizontal R|G|B (cada canal en escala de grises).
    Útil para visualizar los tres canales a la vez.
    """
    if img_bgr is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h = max(r.shape[0], g.shape[0], b.shape[0])
    w = r.shape[1] + g.shape[1] + b.shape[1]
    vis = np.zeros((h, w), dtype=np.uint8)
    x = 0
    for ch in (r, g, b):
        vis[:ch.shape[0], x:x+ch.shape[1]] = ch
        x += ch.shape[1]
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

def mosaic_cmyk(img_bgr):
    """
    Convierte a CMYK (vía PIL) y devuelve mosaico 2x2 (C,M,Y,K) en gris (como BGR).
    """
    if img_bgr is None:
        return None
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    cmyk = pil.convert("CMYK")
    c, m, y, k = [np.array(ch) for ch in cmyk.split()]
    top = np.hstack([c, m])
    bot = np.hstack([y, k])
    grid = np.vstack([top, bot]).astype(np.uint8)
    return cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

def mosaic_hsl(img_bgr):
    """
    Convierte a HLS (OpenCV) y devuelve mosaico 2x2 con H|L en fila 1 y S|negro en fila 2 (como BGR).
    """
    if img_bgr is None:
        return None
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    top = np.hstack([h, l])
    bot = np.hstack([s, np.zeros_like(s)])
    grid = np.vstack([top, bot]).astype(np.uint8)
    return cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

# ========= Histogramas y estadísticas =========
def compute_hist_rgb(img_bgr):
    """Dict {'Rojo': histR, 'Verde': histG, 'Azul': histB} con bins=256."""
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hists = {}
    for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
        datos = img_rgb[:, :, i].ravel()
        hist, _ = np.histogram(datos, bins=256, range=(0, 256))
        hists[canal] = hist
    return hists

def compute_hist_gray(img_bgr):
    """Histograma (bins=256) de la imagen pasada a grises."""
    if img_bgr is None:
        return None
    gray = to_gray_if_needed(img_bgr)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    return hist

def compute_stats_rgb(img_bgr):
    """
    Stats por canal: Energía, Entropía, Asimetría, Media, Varianza.
    Retorna dict {'Rojo': {...}, 'Verde': {...}, 'Azul': {...}}
    """
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = {}
    for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
        datos = img_rgb[:, :, i].ravel()
        hist, _ = np.histogram(datos, bins=256, range=(0, 256))
        s = hist.sum()
        prob = hist / s if s else np.zeros_like(hist, dtype=float)
        energia = float(np.sum(prob ** 2))
        entropia = float(-np.sum([p * log2(p) for p in prob if p > 0]))
        asimetria = float(skew(datos)) if datos.size else 0.0
        media = float(np.mean(datos)) if datos.size else 0.0
        varianza = float(np.var(datos)) if datos.size else 0.0
        res[canal] = {
            'Energía': energia,
            'Entropía': entropia,
            'Asimetría': asimetria,
            'Media': media,
            'Varianza': varianza
        }
    return res

# ========= Binarización =========
def to_binary(img_bgr, thresh=127, use_otsu=False, invert=False):
    """Devuelve imagen binaria 0/255 (uint8)."""
    if img_bgr is None:
        return None
    gray = to_gray_if_needed(img_bgr)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if use_otsu:
        _, binimg = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    else:
        _, binimg = cv2.threshold(gray, int(thresh), 255, flag)
    return binimg

# ========= Etiquetado de Componentes Conectados =========
def ensure_same_size(a, b):
    if a is None or b is None:
        return a, b
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) != (hb, wb):
        b = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_NEAREST)
    return a, b

def label_components(binary_0_255, connectivity=4):
    """labels: int32, num: número de CC."""
    bin01 = (binary_0_255 > 0).astype(np.uint8)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int) if connectivity == 4 else np.ones((3,3), dtype=int)
    labels, num = ndimage.label(bin01, structure=structure)
    return labels, num

def colorize_labels(labels):
    """Mapa RGB (uint8) coloreado por etiqueta (0 = negro)."""
    labels = labels.astype(np.int32)
    k = labels.max()
    if k <= 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(k+1, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[labels]

# ========= Operaciones lógicas y relacionales (sobre binarios) =========
def logic_and(A, B):
    A, B = ensure_same_size(A, B)
    return cv2.bitwise_and(A, B)

def logic_or(A, B):
    A, B = ensure_same_size(A, B)
    return cv2.bitwise_or(A, B)

def logic_xor(A, B):
    A, B = ensure_same_size(A, B)
    return cv2.bitwise_xor(A, B)

def logic_not(img):
    return cv2.bitwise_not(img)

def rel_gt(A, B):
    A, B = ensure_same_size(A, B)
    return np.where(A > B, 255, 0).astype(np.uint8)

def rel_lt(A, B):
    A, B = ensure_same_size(A, B)
    return np.where(A < B, 255, 0).astype(np.uint8)

def rel_eq(A, B):
    A, B = ensure_same_size(A, B)
    return np.where(A == B, 255, 0).astype(np.uint8)

def rel_neq(A, B):
    A, B = ensure_same_size(A, B)
    return np.where(A != B, 255, 0).astype(np.uint8)
