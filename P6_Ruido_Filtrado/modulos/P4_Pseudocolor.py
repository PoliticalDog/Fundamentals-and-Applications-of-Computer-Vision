# modulos/practica4.py
import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ====== Colormaps personalizados ======
_PASTEL_COLORS = [
    (1.0, 0.8, 0.9),  # rosa claro
    (0.8, 1.0, 0.8),  # verde menta
    (0.8, 0.9, 1.0),  # azul lavanda
    (1.0, 1.0, 0.8),  # amarillo suave
    (0.9, 0.8, 1.0)   # violeta claro
]
_PASTEL = LinearSegmentedColormap.from_list("PastelMap", _PASTEL_COLORS, N=256)

CUSTOM_CMAPS = {
    "Pastel (custom)": _PASTEL,  # fijo
}
CUSTOM_LUTS = {}  # se llena al iniciar

def _cmap_to_lut(cmap: LinearSegmentedColormap) -> np.ndarray:
    samples = np.linspace(0.0, 1.0, 256)
    rgba = cmap(samples)              # (256, 4)
    rgb = (rgba[:, :3] * 255).astype(np.uint8)  # (256, 3)
    return rgb

def make_random_colormap(name="RandomMap", n_anchors=5, seed=None) -> LinearSegmentedColormap:
    rng = np.random.default_rng(seed)
    anchors = np.linspace(0.0, 1.0, n_anchors)
    colors = rng.random((n_anchors, 3))
    colors[0]  = colors[0] * 0.2            # más oscuro
    colors[-1] = 0.8 + colors[-1] * 0.2     # más claro
    seg = list(zip(anchors, colors))
    return LinearSegmentedColormap.from_list(name, seg, N=256)

def install_random_cmap(label="Random (custom)", n_anchors=6, seed=None):
    rand_cmap = make_random_colormap("RandomMap", n_anchors=n_anchors, seed=seed)
    CUSTOM_CMAPS[label] = rand_cmap
    CUSTOM_LUTS[label] = _cmap_to_lut(rand_cmap)

# Inicialización de LUTs
CUSTOM_LUTS["Pastel (custom)"] = _cmap_to_lut(CUSTOM_CMAPS["Pastel (custom)"])
install_random_cmap(label="Random (custom)", n_anchors=6, seed=42)  # reproducible al arranque

# ====== Colormaps OpenCV disponibles ======
_COLORMAP_NAMES = {
    "JET": "COLORMAP_JET","HOT": "COLORMAP_HOT","OCEAN": "COLORMAP_OCEAN","PARULA": "COLORMAP_PARULA",
    "RAINBOW": "COLORMAP_RAINBOW","HSV": "COLORMAP_HSV","AUTUMN": "COLORMAP_AUTUMN","BONE": "COLORMAP_BONE",
    "COOL": "COLORMAP_COOL","PINK": "COLORMAP_PINK","SPRING": "COLORMAP_SPRING","SUMMER": "COLORMAP_SUMMER",
    "WINTER": "COLORMAP_WINTER","COPPER": "COLORMAP_COPPER","INFERNO": "COLORMAP_INFERNO",
    "PLASMA": "COLORMAP_PLASMA","MAGMA": "COLORMAP_MAGMA","CIVIDIS": "COLORMAP_CIVIDIS",
    "VIRIDIS": "COLORMAP_VIRIDIS","TWILIGHT": "COLORMAP_TWILIGHT","TURBO": "COLORMAP_TURBO",
}
def build_available_colormaps():
    available = {}
    for name, const in _COLORMAP_NAMES.items():
        if hasattr(cv2, const):
            available[name] = getattr(cv2, const)
    return available

AVAILABLE_COLORMAPS = build_available_colormaps()
GRAYSCALE_OPTION = "Escala de grises (sin pseudocolor)"

# ====== API expuesta a la UI ======
def get_menu_items():
    """Orden: Grises + OpenCV + Custom."""
    return [GRAYSCALE_OPTION] + list(AVAILABLE_COLORMAPS.keys()) + list(CUSTOM_CMAPS.keys())

def regenerate_random():
    """Genera un nuevo 'Random (custom)' con seed=None (distinto cada vez)."""
    install_random_cmap(label="Random (custom)", n_anchors=6, seed=None)

def load_gray(path: str):
    """Carga en escala de grises (uint8) o None si falla."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def apply(gray: np.ndarray, selected: str):
    """
    Devuelve (img_rgb, title_str) según la opción seleccionada.
    - gray: imagen 2D uint8
    - selected: texto del menú
    """
    if selected == GRAYSCALE_OPTION:
        # La UI puede pintar con cmap='gray', aquí devolvemos None para que muestre el gris directo.
        return None, "Escala de grises"

    if selected in AVAILABLE_COLORMAPS:
        code = AVAILABLE_COLORMAPS[selected]
        colored = cv2.applyColorMap(gray, code)             # BGR
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return colored_rgb, f"Pseudocolor (OpenCV): {selected}"

    if selected in CUSTOM_LUTS:
        lut = CUSTOM_LUTS[selected]              # (256, 3) uint8
        colored_rgb = lut[gray]                  # indexación por intensidad
        return colored_rgb, f"Pseudocolor (Custom): {selected}"

    # Fallback
    return None, f"Colormap no disponible: {selected}"
