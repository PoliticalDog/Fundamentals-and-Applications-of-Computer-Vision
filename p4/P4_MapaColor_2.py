import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg") #para tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from tkinter import Tk, Frame, Button, Label, StringVar, OptionMenu, filedialog, BOTH, LEFT, RIGHT, X, TOP

# 1) Mapa de color personalizado fijo (PastelMap)
colorespastel = [
    (1.0, 0.8, 0.9),  # rosa claro
    (0.8, 1.0, 0.8),  # verde menta
    (0.8, 0.9, 1.0),  # azul lavanda
    (1.0, 1.0, 0.8),  # amarillo suave
    (0.9, 0.8, 1.0)   # violeta claro
]
mapapastel = LinearSegmentedColormap.from_list("PastelMap", colorespastel, N=256)

CUSTOM_CMAPS = {
    "Pastel (custom)": mapapastel,  # fijo
}

def cmap_to_lut(cmap: LinearSegmentedColormap) -> np.ndarray:
    samples = np.linspace(0.0, 1.0, 256)
    rgba = cmap(samples)  # (256, 4)
    rgb = (rgba[:, :3] * 255.0).astype(np.uint8)  # (256, 3) en 0..255
    return rgb

CUSTOM_LUTS = {}  # se llena más abajo

# 2) Generador de colormap aleatorio
def make_random_colormap(name="RandomMap", n_anchors=5, seed=None) -> LinearSegmentedColormap:
    rng = np.random.default_rng(seed)
    # Colores aleatorios en [0,1]; forzamos primero y último para cubrir extremos
    anchors = np.linspace(0.0, 1.0, n_anchors)
    colors = rng.random((n_anchors, 3))
    # Opcional: asegurar contraste suficiente en extremos
    colors[0] = colors[0] * 0.2        # más oscuro al inicio
    colors[-1] = 0.8 + colors[-1]*0.2  # más claro al final
    # Empaquetar como lista (pos, color)
    seg = list(zip(anchors, colors))
    return LinearSegmentedColormap.from_list(name, seg, N=256)

def install_random_cmap(label="Random (custom)", n_anchors=5, seed=None):
    rand_cmap = make_random_colormap("RandomMap", n_anchors=n_anchors, seed=seed)
    CUSTOM_CMAPS[label] = rand_cmap
    CUSTOM_LUTS[label] = cmap_to_lut(rand_cmap)

# Inicializamos LUTs de custom conocidos
CUSTOM_LUTS["Pastel (custom)"] = cmap_to_lut(CUSTOM_CMAPS["Pastel (custom)"])
# Creamos por primera vez el aleatorio (semilla fija para tener uno inicial reproducible)
install_random_cmap(label="Random (custom)", n_anchors=6, seed=42)

# 3) Colormaps de OpenCV disponibles
COLORMAP_NAMES = {
    "JET": "COLORMAP_JET",
    "HOT": "COLORMAP_HOT",
    "OCEAN": "COLORMAP_OCEAN",
    "PARULA": "COLORMAP_PARULA",
    "RAINBOW": "COLORMAP_RAINBOW",
    "HSV": "COLORMAP_HSV",
    "AUTUMN": "COLORMAP_AUTUMN",
    "BONE": "COLORMAP_BONE",
    "COOL": "COLORMAP_COOL",
    "PINK": "COLORMAP_PINK",
    "SPRING": "COLORMAP_SPRING",
    "SUMMER": "COLORMAP_SUMMER",
    "WINTER": "COLORMAP_WINTER",
    "COPPER": "COLORMAP_COPPER",
    "INFERNO": "COLORMAP_INFERNO",
    "PLASMA": "COLORMAP_PLASMA",
    "MAGMA": "COLORMAP_MAGMA",
    "CIVIDIS": "COLORMAP_CIVIDIS",
    "VIRIDIS": "COLORMAP_VIRIDIS",
    "TWILIGHT": "COLORMAP_TWILIGHT",
    "TURBO": "COLORMAP_TURBO",
}

def build_available_colormaps():
    available = {}
    for nice_name, const_name in COLORMAP_NAMES.items():
        if hasattr(cv2, const_name):
            available[nice_name] = getattr(cv2, const_name)
    return available

AVAILABLE_COLORMAPS = build_available_colormaps()

GRAYSCALE_OPTION = "Escala de grises (sin pseudocolor)"

# 4) App Tkinter
class PseudocolorApp:
    def __init__(self, master):
        self.master = master
        master.title("Pseudocolor con OpenCV y Colormap personalizado")

        self.image_gray = None
        self.image_path = None

        # Controles superiores
        top = Frame(master)
        top.pack(side=TOP, fill=X, padx=8, pady=8)

        self.btn_select = Button(top, text="Seleccionar imagen…", command=self.select_image)
        self.btn_select.pack(side=LEFT)

        # Opciones del menú: Grises + OpenCV + Custom (Pastel/Random)
        self.cmap_var = StringVar(master)
        self.menu_items = [GRAYSCALE_OPTION] + list(AVAILABLE_COLORMAPS.keys()) + list(CUSTOM_CMAPS.keys())
        self.cmap_var.set(GRAYSCALE_OPTION)
        self.menu = OptionMenu(top, self.cmap_var, *self.menu_items, command=lambda _: self.update_plot())
        self.menu.config(width=32)
        self.menu.pack(side=LEFT, padx=8)

        self.lbl_path = Label(top, text="Sin imagen seleccionada", anchor="w")
        self.lbl_path.pack(side=LEFT, fill=X, expand=True, padx=8)

        # Figura embebida
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # Controles inferiores
        bottom = Frame(master)
        bottom.pack(side=TOP, fill=X, padx=8, pady=8)

        # Botón aplicar
        self.btn_refresh = Button(bottom, text="Aplicar mapa", command=self.update_plot)
        self.btn_refresh.pack(side=RIGHT, padx=4)

        # Botón para generar un nuevo colormap aleatorio (mantiene el mismo nombre de menú)
        self.btn_new_random = Button(bottom, text="Nuevo aleatorio", command=self.generate_new_random)
        self.btn_new_random.pack(side=RIGHT, padx=4)

    def refresh_option_menu(self):
        self.menu["menu"].delete(0, "end")
        self.menu_items = [GRAYSCALE_OPTION] + list(AVAILABLE_COLORMAPS.keys()) + list(CUSTOM_CMAPS.keys())
        for item in self.menu_items:
            self.menu["menu"].add_command(label=item, command=lambda v=item: (self.cmap_var.set(v), self.update_plot()))

    def generate_new_random(self):
        # Cambia los parámetros
        install_random_cmap(label="Random (custom)", n_anchors=6, seed=None)  # seed=None => diferente cada vez
        # refrescar vissta
        if self.cmap_var.get() == "Random (custom)":
            self.update_plot()

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            self.lbl_path.config(text="No se pudo cargar la imagen.")
            return
        self.image_path = path
        self.image_gray = img_gray
        self.lbl_path.config(text=path)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.axis("off")

        if self.image_gray is None:
            self.ax.text(0.5, 0.5, "Selecciona una imagen para visualizar",
                         ha="center", va="center", fontsize=12)
            self.canvas.draw()
            return

        selected = self.cmap_var.get()

        if selected == GRAYSCALE_OPTION:
            self.ax.imshow(self.image_gray, cmap="gray")
            self.ax.set_title("Escala de grises")

        elif selected in AVAILABLE_COLORMAPS:
            # Mapas nativos de OpenCV
            code = AVAILABLE_COLORMAPS[selected]
            colored = cv2.applyColorMap(self.image_gray, code)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            self.ax.imshow(colored_rgb)
            self.ax.set_title(f"Pseudocolor (OpenCV): {selected}")

        elif selected in CUSTOM_LUTS:
            # Mapas personalizados (Matplotlib) usando LUT
            lut = CUSTOM_LUTS[selected]            # (256, 3) uint8
            colored_rgb = lut[self.image_gray]     # indexado por intensidad
            self.ax.imshow(colored_rgb)
            self.ax.set_title(f"Pseudocolor (Custom): {selected}")

        else:
            self.ax.text(0.5, 0.5, f"El colormap '{selected}' no está disponible.",
                         ha="center", va="center", fontsize=12)

        self.fig.tight_layout()
        self.canvas.draw()


# 5) Run
if __name__ == "__main__":
    root = Tk()
    app = PseudocolorApp(root)
    root.mainloop()
