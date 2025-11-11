import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy import ndimage

# ---------- Utilidades ----------
def cv_to_tk(img, max_size=(420, 420)):
    if img is None:
        return None
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_size[0]/w, max_size[1]/h, 1.0)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(rgb))

def to_binary(img, thresh=127, use_otsu=True, invert=False):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if use_otsu:
        _, binimg = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    else:
        _, binimg = cv2.threshold(gray, int(thresh), 255, flag)
    return binimg

def label_components(binary_0_255, connectivity=4):
    bin01 = (binary_0_255 > 0).astype(np.uint8)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int) if connectivity == 4 else np.ones((3,3), dtype=int)
    labels, num = ndimage.label(bin01, structure=structure)
    return labels, num

def colorize_labels(labels):
    labels = labels.astype(np.int32)
    k = labels.max()
    if k <= 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(k+1, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[labels]

# ---------- App ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Práctica 3 - 4 vs 8")
        root.geometry("980x560")

        # Mini tema (Nord)
        self.colors = {
            'bg_main': '#2E3440','bg_secondary': '#3B4252','bg_accent': '#4C566A',
            'text_primary': '#ECEFF4','text_secondary': '#D8DEE9','button_primary': '#5E81AC'
        }
        root.configure(bg=self.colors['bg_main'])
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors['bg_main'])
        style.configure("TLabel", background=self.colors['bg_main'], foreground=self.colors['text_primary'])
        style.configure("TLabelframe", background=self.colors['bg_secondary'], foreground=self.colors['text_primary'])
        style.configure("TButton", padding=6, font=("Segoe UI", 9, "bold"))

        # Estado
        self.img = None
        self.binimg = None
        self.labels4 = None
        self.labels8 = None
        self.color4 = None
        self.color8 = None
        self.num4 = 0
        self.num8 = 0

        # ---- Layout
        left = tk.Frame(root, bg=self.colors['bg_secondary'], bd=2, relief="groove")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right = tk.Frame(root, bg=self.colors['bg_main'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Controles
        ttk.Label(left, text="Flujo: Cargar → Binarizar → Etiquetar → Mostrar",
                  foreground=self.colors['text_secondary']).pack(anchor="w", pady=(4,8))

        #cargar imagen
        ttk.Button(left, text="Cargar imagen…", command=self.load_image).pack(fill=tk.X, pady=2)

        # Binarización
        binf = tk.LabelFrame(left, text="Binarización", bg=self.colors['bg_secondary'], fg=self.colors['text_primary'])
        binf.pack(fill=tk.X, pady=8)
        self.var_otsu = tk.BooleanVar(value=True)
        self.var_invert = tk.BooleanVar(value=False)
        self.th = tk.IntVar(value=127)
        ttk.Checkbutton(binf, text="Otsu (auto)", variable=self.var_otsu).pack(anchor="w", padx=6, pady=(4,0))
        ttk.Checkbutton(binf, text="Invertir (fondo↔objeto)", variable=self.var_invert).pack(anchor="w", padx=6)
        tk.Scale(binf, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.th,
                 bg=self.colors['bg_secondary'], highlightthickness=0,
                 troughcolor=self.colors['bg_accent']).pack(fill=tk.X, padx=6, pady=(2,4))
        ttk.Button(binf, text="Aplicar binarización", command=self.apply_binary).pack(fill=tk.X, padx=6, pady=(0,4))

        # Etiquetado
        labf = tk.LabelFrame(left, text="Etiquetado", bg=self.colors['bg_secondary'], fg=self.colors['text_primary'])
        labf.pack(fill=tk.X, pady=8)
        ttk.Button(labf, text="Etiquetar (4 y 8)", command=self.apply_labeling).pack(fill=tk.X, padx=6, pady=4)

        # Mostrar
        showf = tk.LabelFrame(left, text="Mostrar", bg=self.colors['bg_secondary'], fg=self.colors['text_primary'])
        showf.pack(fill=tk.X, pady=8)
        ttk.Button(showf, text="Original", command=self.show_original).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(showf, text="Binaria", command=self.show_binary).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(showf, text="4-Conex", command=self.show_labels4).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(showf, text="8-Conex", command=self.show_labels8).pack(fill=tk.X, padx=6, pady=2)


        # Visor derecho (una sola imagen a la vez)
        self.viewer = tk.Label(right, text="Aquí se muestra la imagen", bg=self.colors['bg_accent'],
                               fg=self.colors['text_primary'], width=60, height=28)
        self.viewer.pack(expand=True, fill=tk.BOTH)
        self.tkimg = None

    # ---- Acciones
    def load_image(self):
        path = filedialog.askopenfilename(title="Selecciona imagen",
                                          filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
           messagebox.showerror("Error", "No se pudo cargar la imagen.")
           return
        self.img = img
        self.binimg = None
        self.labels4 = self.labels8 = None
        self.color4 = self.color8 = None
        self.num4 = self.num8 = 0
      
        self.show_on_viewer(self.img)

    def apply_binary(self):
        if self.img is None:
            return messagebox.showinfo("Atención", "Carga una imagen primero.")
        self.binimg = to_binary(self.img, thresh=self.th.get(),
                                use_otsu=self.var_otsu.get(), invert=self.var_invert.get())
      
        self.show_on_viewer(self.binimg)

    def apply_labeling(self):
        if self.binimg is None:
            return messagebox.showinfo("Atención", "Binariza primero.")
        self.labels4, self.num4 = label_components(self.binimg, connectivity=4)
        self.labels8, self.num8 = label_components(self.binimg, connectivity=8)
        self.color4 = colorize_labels(self.labels4)
        self.color8 = colorize_labels(self.labels8)
        # Imprimir en consola (como pediste)
        print("===== RESULTADOS DE ETIQUETADO =====")
        print(f"Objetos (vecindad 4): {self.num4}")
        print(f"Objetos (vecindad 8): {self.num8}")
        print("====================================")

    # ---- Mostrar
    def show_on_viewer(self, img):
        self.tkimg = cv_to_tk(img)
        if self.tkimg is not None:
            self.viewer.configure(image=self.tkimg, text="")
        else:
            self.viewer.configure(image="", text="(sin imagen)")

    def show_original(self):
        if self.img is None:
            return messagebox.showinfo("Info", "No hay imagen cargada.")
        self.show_on_viewer(self.img)

    def show_binary(self):
        if self.binimg is None:
            return messagebox.showinfo("Info", "Aún no has binarizado.")
        self.show_on_viewer(self.binimg)

    def show_labels4(self):
        if self.color4 is None:
            return messagebox.showinfo("Info", "Aún no has etiquetado (4 y 8).")
        self.show_on_viewer(self.color4)

    def show_labels8(self):
        if self.color8 is None:
            return messagebox.showinfo("Info", "Aún no has etiquetado (4 y 8).")
        self.show_on_viewer(self.color8)

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
