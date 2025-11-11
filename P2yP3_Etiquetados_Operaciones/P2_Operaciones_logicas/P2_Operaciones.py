import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ===== Crear carpeta 'out' si no existe =====
os.makedirs("out", exist_ok=True)

# ===== Utilidades =====
def cv_to_tk(img_bgr, max_size=(380, 380)):
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(max_size[0]/w, max_size[1]/h, 1.0)
    if scale != 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil_img)

def ensure_same_size(a, b):
    if a is None or b is None:
        return a, b
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) != (hb, wb):
        b = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_NEAREST)
    return a, b

def to_binary(img, thresh=127, use_otsu=False, invert=False):
    if img is None:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if use_otsu:
        _, binarized = cv2.threshold(gray, 0, 255, ttype + cv2.THRESH_OTSU)
    else:
        _, binarized = cv2.threshold(gray, int(thresh), 255, ttype)
    return binarized

# ===== Aplicación =====
class App:
    def __init__(self, root):
        self.root = root
        root.title("🧠 Práctica 2 - Operaciones Lógicas y Relacionales")
        root.geometry("1250x650")

        # ==== Tema de colores ====
        self.colors = {
            'bg_main': '#2E3440',
            'bg_secondary': '#3B4252',
            'bg_accent': '#4C566A',
            'text_primary': '#ECEFF4',
            'text_secondary': '#D8DEE9',
            'button_primary': '#5E81AC',
            'button_success': '#A3BE8C',
            'button_warning': '#EBCB8B',
            'button_danger': '#BF616A',
            'button_info': '#88C0D0',
            'button_accent': '#81A1C1',
            'accent': '#81A1C1',
            'border': '#434C5E'
        }
        root.configure(bg=self.colors['bg_main'])

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("TLabel", background=self.colors['bg_main'], foreground=self.colors['text_primary'])
        style.configure("TLabelframe", background=self.colors['bg_secondary'], foreground=self.colors['text_primary'])
        style.configure("TFrame", background=self.colors['bg_main'])
        style.configure("Horizontal.TScale", background=self.colors['bg_main'])
        style.configure("Accent.TButton", background=self.colors['button_primary'], foreground="white")
        style.map("Accent.TButton", background=[("active", self.colors['button_info'])])

        self.imgA = None
        self.imgB = None
        self.binA = None
        self.binB = None
        self.result = None

        # ===== Layout =====
        left = tk.Frame(root, bg=self.colors['bg_secondary'], bd=2, relief="groove")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right = tk.Frame(root, bg=self.colors['bg_main'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # ==== Panel izquierdo ====
        ttk.Label(left, text="IMAGEN A", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,2))
        ttk.Button(left, text="Cargar Imagen 1", command=self.load_A, style="Accent.TButton").pack(fill=tk.X, pady=2)

        self.thA = tk.IntVar(value=127)
        self.otsuA = tk.BooleanVar(value=False)
        ttk.Label(left, text="Umbral A").pack(anchor="w", pady=(6,0))
        tk.Scale(left, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.thA,
                 bg=self.colors['bg_secondary'], troughcolor=self.colors['accent']).pack(fill=tk.X)
        tk.Checkbutton(left, text="Usar Otsu", variable=self.otsuA, bg=self.colors['bg_secondary'],
                       fg=self.colors['text_secondary']).pack(anchor="w")
        ttk.Button(left, text="Binarizar A", command=self.binarize_A).pack(fill=tk.X, pady=4)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="IMAGEN B", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4,2))
        ttk.Button(left, text="Cargar Imagen 2", command=self.load_B).pack(fill=tk.X, pady=2)

        self.thB = tk.IntVar(value=127)
        self.otsuB = tk.BooleanVar(value=False)
        ttk.Label(left, text="Umbral B").pack(anchor="w", pady=(6,0))
        tk.Scale(left, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.thB,
                 bg=self.colors['bg_secondary'], troughcolor=self.colors['accent']).pack(fill=tk.X)
        tk.Checkbutton(left, text="Usar Otsu", variable=self.otsuB, bg=self.colors['bg_secondary'],
                       fg=self.colors['text_secondary']).pack(anchor="w")
        ttk.Button(left, text="Binarizar B", command=self.binarize_B).pack(fill=tk.X, pady=4)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # === Operaciones lógicas ===
        lf = tk.LabelFrame(left, text="Operaciones Lógicas", bg=self.colors['bg_secondary'],
                           fg=self.colors['text_primary'], font=("Segoe UI", 9, "bold"))
        lf.pack(fill=tk.X, pady=4)
        ttk.Button(lf, text="AND",   command=self.op_and).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="OR",    command=self.op_or).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="XOR",   command=self.op_xor).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="NOT A", command=self.op_not_a).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="NOT B", command=self.op_not_b).pack(fill=tk.X, padx=4, pady=2)

        # === Operaciones relacionales ===
        rf = tk.LabelFrame(left, text="Operaciones Relacionales", bg=self.colors['bg_secondary'],
                           fg=self.colors['text_primary'], font=("Segoe UI", 9, "bold"))
        rf.pack(fill=tk.X, pady=4)
        ttk.Button(rf, text="A > B",   command=self.op_gt).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(rf, text="A < B",   command=self.op_lt).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(rf, text="A == B",  command=self.op_eq).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(rf, text="A != B",  command=self.op_neq).pack(fill=tk.X, padx=4, pady=2)

        # ==== Panel derecho ====
        grids = tk.Frame(right, bg=self.colors['bg_main'])
        grids.pack(expand=True, fill=tk.BOTH)

        self.lblA = tk.Label(grids, text="A", bg=self.colors['bg_accent'], fg=self.colors['text_primary'], width=50, height=22)
        self.lblB = tk.Label(grids, text="B", bg=self.colors['bg_accent'], fg=self.colors['text_primary'], width=50, height=22)
        self.lblR = tk.Label(grids, text="Resultado", bg=self.colors['bg_accent'], fg=self.colors['text_primary'], width=50, height=22)

        self.lblA.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.lblB.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        self.lblR.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="nsew")

        grids.grid_columnconfigure(0, weight=1)
        grids.grid_columnconfigure(1, weight=1)
        grids.grid_rowconfigure(0, weight=1)
        grids.grid_rowconfigure(1, weight=1)

        self.tkA = self.tkB = self.tkR = None

    # ==== Métodos de carga y binarización ====
    def load_A(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.imgA = cv2.imread(path)
        self.binA = None
        self.update_previewA()

    def load_B(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.imgB = cv2.imread(path)
        self.binB = None
        self.update_previewB()

    def binarize_A(self):
        if self.imgA is None:
            return messagebox.showinfo("Atención", "Carga primero la imagen A.")
        self.binA = to_binary(self.imgA, self.thA.get(), self.otsuA.get())
        self.update_previewA()

    def binarize_B(self):
        if self.imgB is None:
            return messagebox.showinfo("Atención", "Carga primero la imagen B.")
        self.binB = to_binary(self.imgB, self.thB.get(), self.otsuB.get())
        self.update_previewB()

    def update_previewA(self):
        img = self.binA if self.binA is not None else self.imgA
        self.tkA = cv_to_tk(img)
        if self.tkA: self.lblA.configure(image=self.tkA, text="")

    def update_previewB(self):
        img = self.binB if self.binB is not None else self.imgB
        self.tkB = cv_to_tk(img)
        if self.tkB: self.lblB.configure(image=self.tkB, text="")

    def show_result(self, img):
        self.result = img
        self.tkR = cv_to_tk(img)
        if self.tkR: self.lblR.configure(image=self.tkR, text="")

    # ==== Guardado de resultados ====
    def save_result(self, name):
        """Guarda el resultado en formato PNG dentro de la carpeta 'out'."""
        if self.result is None:
            return
        filename = f"out/{name}.png"
        cv2.imwrite(filename, self.result)
        messagebox.showinfo("Guardado", f"Resultado guardado como:\n{filename}")

    # ==== Operaciones lógicas y relacionales ====
    def get_pair_binary(self):
        if self.binA is None or self.binB is None:
            messagebox.showinfo("Atención", "Binariza ambas imágenes primero.")
            return None, None
        A, B = ensure_same_size(self.binA, self.binB)
        return A, B

    def op_and(self):
        A, B = self.get_pair_binary()
        if A is None: return
        self.show_result(cv2.bitwise_and(A, B))
        self.save_result("AND")

    def op_or(self):
        A, B = self.get_pair_binary()
        if A is None: return
        self.show_result(cv2.bitwise_or(A, B))
        self.save_result("OR")

    def op_xor(self):
        A, B = self.get_pair_binary()
        if A is None: return
        self.show_result(cv2.bitwise_xor(A, B))
        self.save_result("XOR")

    def op_not_a(self):
        if self.binA is None:
            return messagebox.showinfo("Atención", "Binariza primero la imagen A.")
        self.show_result(cv2.bitwise_not(self.binA))
        self.save_result("NOT_A")

    def op_not_b(self):
        if self.binB is None:
            return messagebox.showinfo("Atención", "Binariza primero la imagen B.")
        self.show_result(cv2.bitwise_not(self.binB))
        self.save_result("NOT_B")

    def op_gt(self):
        A, B = self.get_pair_binary()
        if A is None: return
        self.show_result(np.where(A > B, 255, 0).astype(np.uint8))
        self.save_result("A_GT_B")

    def op_lt(self):
        A, B = self.get_pair_binary()
        if A is None: return
        self.show_result(np.where(A < B, 255, 0).astype(np.uint8))
        self.save_result("A_LT_B")

    def op_eq(self):
        A, B = self.get_pair_binary()
        if A is None: return
        self.show_result(np.where(A == B, 255, 0).astype(np.uint8))
        self.save_result("A_EQ_B")

    def op_neq(self):
        A, B = self.get_pair_binary()
        if A is None: return
        self.show_result(np.where(A != B, 255, 0).astype(np.uint8))
        self.save_result("A_NEQ_B")

# ==== Inicio del programa ====
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
