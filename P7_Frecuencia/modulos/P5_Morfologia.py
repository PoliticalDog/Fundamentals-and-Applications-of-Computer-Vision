# archivo: modulos/practica5.py
import cv2
import numpy as np
from typing import Literal, Optional, Tuple

# ===================== METODOS =====================
#Gries 
def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convierte BGR a GRAY si es necesario; preserva si ya es GRAY."""
    if img is None:
        raise ValueError("Imagen None.")
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

#bINARIOS
def to_binary(img: np.ndarray, thresh: int = 127, use_otsu: bool = False, invert: bool = False) -> np.ndarray:
    """
    Convierte la imagen a binaria (0/255) con umbral fijo u Otsu.
    """
    gray = _to_gray(img)
    if use_otsu:
        ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binimg = cv2.threshold(gray, 0, 255, ttype | cv2.THRESH_OTSU)
    else:
        t = np.clip(int(thresh), 0, 255)
        ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binimg = cv2.threshold(gray, t, 255, ttype)
    return binimg

#Redimensionar imagenes
def ensure_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona B a A si no coinciden tamaños (interpolación adecuada).
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) == (hb, wb):
        return a, b
    interp = cv2.INTER_NEAREST if (len(b.shape) == 2 or b.shape[2] == 1) else cv2.INTER_AREA
    b2 = cv2.resize(b, (wa, ha), interpolation=interp)
    return a, b2

#Construe ell amepo de EE
def make_se(ksize: int = 3, shape: Literal["rect", "ellipse", "cross"] = "rect") -> np.ndarray:
    """
    Crea elemento estructurante (EE) con forma y tamaño dados.
    """
    k = max(1, int(ksize))
    if k % 2 == 0:  # forzar impar en muchos casos morfológicos
        k += 1
    if shape == "ellipse":
        st = cv2.MORPH_ELLIPSE
    elif shape == "cross":
        st = cv2.MORPH_CROSS
    else:
        st = cv2.MORPH_RECT
    return cv2.getStructuringElement(st, (k, k))

#Validacion bianria
def _is_binary(img: np.ndarray) -> bool:
    """Heurística: ¿solo 0/255?"""
    u = np.unique(img)
    return u.size <= 2 and set(u.tolist()).issubset({0, 255})

# ================= Operaciones básicas (binario y gris) =================
#Erocion
def erode(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    return cv2.erode(img, make_se(ksize, shape), iterations=iterations)

#Dilatacion
def dilate(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    return cv2.dilate(img, make_se(ksize, shape), iterations=iterations)

#Apertura
def open_morph(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, make_se(ksize, shape), iterations=iterations)

#Cierre
def close_morph(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, make_se(ksize, shape), iterations=iterations)

# ===================== Gradientes morfológicos =====================
#Gradiante
def gradient(img: np.ndarray, ksize: int = 3, shape: str = "rect",
             mode: Literal["sym", "int", "ext"] = "sym") -> np.ndarray:
    """
    Gradientes:
      - 'sym'  (simétrico): dilate - erode
      - 'int'  (interno):   img - erode
      - 'ext'  (externo):   dilate - img
    Soporta gris/binario.
    """
    se = make_se(ksize, shape)
    if mode == "sym":
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)
    elif mode == "int":
        ero = cv2.erode(img, se)
        return cv2.subtract(img, ero)
    else:  # "ext"
        dil = cv2.dilate(img, se)
        return cv2.subtract(dil, img)

# ===================== Top-Hat / Black-Hat =====================

def top_hat(img: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """Resalta regiones más claras que su vecindad."""
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, make_se(ksize, shape))

def black_hat(img: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """Resalta regiones más oscuras que su vecindad."""
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, make_se(ksize, shape))

# ===================== Frontera (boundary) =====================

def boundary(img: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """
    Frontera interna clásica: B(img) = img - erode(img).
    Para binario: devuelve contorno en 0/255.
    Para gris: resalta bordes finos.
    """
    se = make_se(ksize, shape)
    ero = cv2.erode(img, se)
    return cv2.subtract(img, ero)

# ===================== Hit-or-Miss (binario) =====================

def hit_or_miss(binimg: np.ndarray,
                kernel_hit: np.ndarray,
                kernel_miss: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Transformada Hit-or-Miss para BINARIO (0/255).
    kernel_hit  : EE con 1s donde se exige 1 (255) y 0 donde 'no importa'.
    kernel_miss : EE con 1s donde se exige 0 y 0 donde 'no importa'.
                  Si es None, se calcula como complemento de kernel_hit (aprox).
    OpenCV espera binario en 0/1; convertimos y devolvemos 0/255.
    """
    img = _to_gray(binimg)
    if not _is_binary(img):
        raise ValueError("hit_or_miss requiere imagen binaria (0/255).")
    img01 = (img > 0).astype(np.uint8)

    kh = (kernel_hit > 0).astype(np.uint8)
    if kernel_miss is None:
        # Aproximación: donde kh == 0, asumimos 'miss' (exige 0)
        km = (kh == 0).astype(np.uint8)
    else:
        km = (kernel_miss > 0).astype(np.uint8)

    # OpenCV usa 1,0,-1 cuando se arma con MORPH_HITMISS y máscaras ternarias;
    # aquí componemos hit-or-miss vía intersección: (img ⊖ kh) ∩ (~img ⊖ km)
    hit  = cv2.erode(img01, kh)
    miss = cv2.erode(1 - img01, km)
    out01 = cv2.bitwise_and(hit, miss)
    return (out01 * 255).astype(np.uint8)

# ===================== Adelgazamiento (Thinning) =====================

def thinning(binimg: np.ndarray, max_iters: int = 0) -> np.ndarray:
    """
    Adelgazamiento (Zhang–Suen) sobre binario 0/255.
    max_iters=0 => itera hasta convergencia.
    Devuelve 0/255.
    """
    img = _to_gray(binimg)
    if not _is_binary(img):
        raise ValueError("thinning requiere imagen binaria (0/255).")
    # Convertir a 0/1
    th = (img > 0).astype(np.uint8)

    def neighbors(y, x):
        # p2..p9 en sentido horario empezando arriba (8-neighbors)
        return [th[y-1, x], th[y-1, x+1], th[y, x+1], th[y+1, x+1],
                th[y+1, x], th[y+1, x-1], th[y, x-1], th[y-1, x-1]]

    def transitions(nb):
        # número de transiciones 0->1 en la secuencia circular
        return sum((nb[i] == 0 and nb[(i+1) % 8] == 1) for i in range(8))

    changed = True
    iters = 0
    h, w = th.shape
    while changed and (max_iters == 0 or iters < max_iters):
        changed = False
        m1 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                p = th[y, x]
                if p != 1:
                    continue
                nb = neighbors(y, x)
                C = transitions(nb)
                N = sum(nb)
                if 2 <= N <= 6 and C == 1 and (nb[0]*nb[2]*nb[4] == 0) and (nb[2]*nb[4]*nb[6] == 0):
                    m1.append((y, x))
        if m1:
            changed = True
            for (y, x) in m1:
                th[y, x] = 0

        m2 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                p = th[y, x]
                if p != 1:
                    continue
                nb = neighbors(y, x)
                C = transitions(nb)
                N = sum(nb)
                if 2 <= N <= 6 and C == 1 and (nb[0]*nb[2]*nb[6] == 0) and (nb[0]*nb[4]*nb[6] == 0):
                    m2.append((y, x))
        if m2:
            changed = True
            for (y, x) in m2:
                th[y, x] = 0

        iters += 1

    return (th * 255).astype(np.uint8)

# ===================== Esqueleto Morfológico =====================

def skeletonize(binimg: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """
    Esqueleto morfológico por iteración: 
      S = ⋃ (Erode^k(img) - Open(Erode^k(img)))
    hasta que Erode^k(img) sea vacío.
    Entrada: binario 0/255. Salida: 0/255.
    """
    img = _to_gray(binimg)
    if not _is_binary(img):
        raise ValueError("skeletonize requiere imagen binaria (0/255).")

    se = make_se(ksize, shape)
    skel = np.zeros_like(img)
    eroded = img.copy()

    while True:
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, se)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        eroded = cv2.erode(eroded, se)
        if cv2.countNonZero(eroded) == 0:
            break
    return skel

# ===================== Suavizado morfológico =====================

def smooth(img: np.ndarray, ksize: int = 3, shape: str = "rect",
           passes: int = 1, mode: Literal["open_close", "close_open"] = "open_close") -> np.ndarray:
    """
    Suavizado morfológico para gris/binario:
      - open_close: apertura seguida de cierre (elimina ruido claro y rellena huecos pequeños)
      - close_open: cierre seguido de apertura (inverso)
    """
    out = img.copy()
    for _ in range(max(1, int(passes))):
        if mode == "open_close":
            out = open_morph(out, ksize, shape)
            out = close_morph(out, ksize, shape)
        else:
            out = close_morph(out, ksize, shape)
            out = open_morph(out, ksize, shape)
    return out

# ===================== Atajos “tradicionales” (combinando básicas) =====================

def apertura_tradicional(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Apertura = erosión seguida de dilatación (implementación 'manual')."""
    se = make_se(ksize, shape)
    out = cv2.erode(img, se, iterations=iterations)
    out = cv2.dilate(out, se, iterations=iterations)
    return out

def cierre_tradicional(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Cierre = dilatación seguida de erosión (implementación 'manual')."""
    se = make_se(ksize, shape)
    out = cv2.dilate(img, se, iterations=iterations)
    out = cv2.erode(out, se, iterations=iterations)
    return out

# ===================== Utilidades de guardado opcional =====================

def save(path: str, img: np.ndarray) -> None:
    """Guarda imagen con cv2.imwrite; levanta excepción si falla."""
    if not cv2.imwrite(path, img):
        raise IOError(f"No se pudo guardar: {path}")

# ===================== Ejemplos rápidos (opcional) =====================

if __name__ == "__main__":
    # Ejemplo mínimo de uso (ajusta rutas):
    example_path = None  # "ejemplos/monedas.png"
    if example_path:
        src = cv2.imread(example_path)
        gray = _to_gray(src)
        bin_img = to_binary(gray, use_otsu=True)

        k = 3
        se_shape = "rect"

        ero = erode(gray, k, se_shape)
        dil = dilate(gray, k, se_shape)
        opn = open_morph(gray, k, se_shape)
        cls = close_morph(gray, k, se_shape)
        gsym = gradient(gray, k, se_shape, "sym")
        gint = gradient(gray, k, se_shape, "int")
        gext = gradient(gray, k, se_shape, "ext")
        th  = top_hat(gray, k, se_shape)
        bh  = black_hat(gray, k, se_shape)
        bnd = boundary(bin_img, k, se_shape)
        ske = skeletonize(bin_img, k, se_shape)
        thi = thinning(bin_img)

        # Guarda si quieres verificar:
        cv2.imwrite("out_p5/erosion.png", ero)
        cv2.imwrite("out_p5/dilatacion.png", dil)
        cv2.imwrite("out_p5/apertura.png", opn)
        cv2.imwrite("out_p5/cierre.png", cls)
        cv2.imwrite("out_p5/gradiente_sym.png", gsym)
        cv2.imwrite("out_p5/gradiente_int.png", gint)
        cv2.imwrite("out_p5/gradiente_ext.png", gext)
        cv2.imwrite("out_p5/top_hat.png", th)
        cv2.imwrite("out_p5/black_hat.png", bh)
        cv2.imwrite("out_p5/frontera.png", bnd)
        cv2.imwrite("out_p5/esqueleto.png", ske)
        cv2.imwrite("out_p5/adelgazamiento.png", thi)
