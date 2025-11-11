# interfaz.py
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modulos import P1_Histograma_RGB_BIN as p1
from modulos import P2_Logicas_Relacionales as p2
from modulos import P3_Etiquetado_4_8 as p3
from modulos import P4_Pseudocolor as p4
from modulos import P5_Morfologia as p5

# ========= Helpers de UI =========
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

THEME = {
    'bg_main': '#2E3440','bg_secondary': '#3B4252','bg_accent': '#4C566A',
    'text_primary': '#ECEFF4','text_secondary': '#D8DEE9',
    'button_primary': '#5E81AC','button_success': '#A3BE8C',
    'button_warning': '#EBCB8B','button_danger': '#BF616A',
    'button_info': '#88C0D0','button_accent': '#81A1C1','accent':'#81A1C1',
    'border':'#434C5E'
}

# ===== ScrollableFrame reutilizable (Canvas + Scrollbar) =====
class ScrollableFrame(tk.Frame):
    def __init__(self, master, bg="#ffffff", width=300, *args, **kwargs):
        super().__init__(master, bg=bg, *args, **kwargs)
        # Canvas + barra vertical
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame interior que contendrá tus controles
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Ajuste de scrollregion y sincronización de ancho
        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Soporte de rueda del mouse (Windows/Mac/Linux)
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Mantiene el frame interior con el mismo ancho visual del canvas
        self.canvas.itemconfigure(self._win, width=event.width)

    # ---- Mousewheel cross-platform ----
    def _on_mousewheel_windows_mac(self, event):
        # Windows: event.delta ±120 ; macOS: ±1 (a veces ±120 también)
        step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step, "units")

    def _on_mousewheel_linux_up(self, event):
        self.canvas.yview_scroll(-1, "units")

    def _on_mousewheel_linux_down(self, event):
        self.canvas.yview_scroll(1, "units")

    def _bind_mousewheel(self, _):
        # Windows / Mac
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows_mac)
        # Linux (X11)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux_up)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux_down)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

# ========= Pestaña Práctica 1 =========
class TabPractica1(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=THEME['bg_main'])
        self.ruta = None
        self.img_pil = None
        self.img_gray = None
        self._build()

    def _build(self):
        # Lado izquierdo con scroll
        sf = ScrollableFrame(self, bg=THEME['bg_secondary'])
        sf.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=8)
        left = sf.inner

        right = tk.Frame(self, bg=THEME['bg_main'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=8, pady=8)

        # Botones izquierda
        ttk.Style().theme_use("clam")
        ttk.Button(left, text="Cargar imagen", command=self.load_image).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Separar RGB", command=self.separar_rgb).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="A grises", command=self.to_gray).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Binarizar (umbral)", command=self.binarizar_umbral).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Binarizar (Otsu)", command=self.binarizar_otsu).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Convertir a CMYK", command=self.to_cmyk).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Convertir a HSL", command=self.to_hsl).pack(fill=tk.X, pady=3)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        tk.Label(left, text="Histogramas", bg=THEME['bg_secondary'], fg=THEME['text_primary']).pack()
        ttk.Button(left, text="Rojo", command=lambda: self.plot_hist('Rojo')).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Verde", command=lambda: self.plot_hist('Verde')).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Azul", command=lambda: self.plot_hist('Azul')).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Todos", command=lambda: self.plot_hist('Todos')).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Grises", command=lambda: self.plot_hist('Grises')).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Calcular stats (RGB/Gris)", command=self.calc_stats_console).pack(fill=tk.X, pady=2)

        # Panel derecho
        self.preview = tk.Label(right, text="(preview)", bg=THEME['bg_accent'], fg=THEME['text_primary'], height=18)
        self.preview.pack(side=tk.TOP, fill=tk.X, pady=4)

        self.fig = Figure(figsize=(6,4), facecolor=THEME['bg_main'])
        self.ax = self.fig.add_subplot(111, facecolor=THEME['bg_secondary'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not path: return
        self.ruta = path
        try:
            self.img_pil = p1.cargar_imagen(path)  # también muestra la imagen con plt (se respeta)
            # preview embebido
            prev = self.img_pil.copy()
            prev.thumbnail((360, 360))
            self.tkprev = ImageTk.PhotoImage(prev)
            self.preview.configure(image=self.tkprev, text="")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def separar_rgb(self):
        if not self.img_pil:
            return messagebox.showinfo("Aviso", "Carga una imagen.")
        p1.separar_rgb(self.img_pil)

    def to_gray(self):
        if not self.img_pil:
            return messagebox.showinfo("Aviso", "Carga una imagen.")
        self.img_gray = p1.convertir_a_grises(self.img_pil)
        try:
            img_pil = Image.fromarray(self.img_gray)
            prev = img_pil.copy(); prev.thumbnail((360,360))
            self.tkprev = ImageTk.PhotoImage(prev)
            self.preview.configure(image=self.tkprev, text="")
        except:
            pass

    def binarizar_umbral(self):
        if self.img_gray is None:
            if not self.img_pil: return messagebox.showinfo("Aviso", "Carga una imagen.")
            self.img_gray = p1.convertir_a_grises(self.img_pil)
        umbral = simpledialog.askinteger("Umbral", "Valor (0-255):", minvalue=0, maxvalue=255)
        if umbral is None: return
        p1.binarizar_imagen_umbral(self.img_gray, umbral)

    def binarizar_otsu(self):
        if self.img_gray is None:
            if not self.img_pil: return messagebox.showinfo("Aviso", "Carga una imagen.")
            self.img_gray = p1.convertir_a_grises(self.img_pil)
        p1.binarizar_imagen_otsu(self.img_gray)

    def to_cmyk(self):
        if not self.img_pil: return messagebox.showinfo("Aviso", "Carga una imagen.")
        p1.convertir_a_cmyk(self.img_pil)

    def to_hsl(self):
        if not self.img_pil: return messagebox.showinfo("Aviso", "Carga una imagen.")
        p1.convertir_a_hsl(self.img_pil)

    def plot_hist(self, canal):
        if not self.ruta: return messagebox.showinfo("Aviso", "Carga una imagen.")
        self.ax.clear()
        self.ax.set_facecolor(THEME['bg_secondary'])
        self.ax.grid(True, color=THEME['text_secondary'], alpha=0.25)

        if canal in ("Rojo","Verde","Azul"):
            h = p1.compute_histogramas_rgb_arrays(self.ruta)
            if h is None: return
            color = {'Rojo':'red','Verde':'green','Azul':'blue'}[canal]
            self.ax.plot(h[canal], color=color, label=canal)
            self.ax.set_title(f"Histograma - {canal}", color=THEME['text_primary'])
        elif canal == "Todos":
            h = p1.compute_histogramas_rgb_arrays(self.ruta)
            if h is None: return
            self.ax.plot(h['Rojo'],  color='red',   label='Rojo')
            self.ax.plot(h['Verde'], color='green', label='Verde')
            self.ax.plot(h['Azul'],  color='blue',  label='Azul')
            self.ax.legend()
            self.ax.set_title("Histogramas RGB", color=THEME['text_primary'])
        elif canal == "Grises":
            hg = p1.compute_histograma_gris_array(self.ruta)
            if hg is None: return
            self.ax.plot(hg, color='gray')
            self.ax.set_title("Histograma - Grises", color=THEME['text_primary'])

        self.ax.set_xlabel("Intensidad", color=THEME['text_primary'])
        self.ax.set_ylabel("Frecuencia", color=THEME['text_primary'])
        for sp in self.ax.spines.values():
            sp.set_color(THEME['text_secondary'])
        self.canvas.draw()

    def calc_stats_console(self):
        if not self.ruta: return messagebox.showinfo("Aviso", "Carga una imagen.")
        p1.calcular_histogramas_rgb(self.ruta)
        p1.calcular_histograma_grises(self.ruta)


# ========= Pestaña Práctica 2 =========
class TabPractica2(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=THEME['bg_main'])
        self.imgA = None; self.imgB = None
        self.binA = None; self.binB = None
        self.result = None
        os.makedirs("out", exist_ok=True)
        self._build()

    def _build(self):
        # Lado izquierdo con scroll
        sf = ScrollableFrame(self, bg=THEME['bg_secondary'])
        sf.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=8)
        left = sf.inner

        right = tk.Frame(self, bg=THEME['bg_main'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=8, pady=8)

        ttk.Style().theme_use("clam")
        ttk.Label(left, text="IMAGEN A").pack(anchor="w")
        ttk.Button(left, text="Cargar A", command=self.load_A).pack(fill=tk.X, pady=2)
        self.thA = tk.IntVar(value=127); self.otsuA = tk.BooleanVar(value=False)
        ttk.Label(left, text="Umbral A").pack(anchor="w")
        tk.Scale(left, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.thA,
                 bg=THEME['bg_secondary'], troughcolor=THEME['accent']).pack(fill=tk.X)
        ttk.Checkbutton(left, text="Usar Otsu", variable=self.otsuA).pack(anchor="w")
        ttk.Button(left, text="Binarizar A", command=self.binarize_A).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(left, text="IMAGEN B").pack(anchor="w")
        ttk.Button(left, text="Cargar B", command=self.load_B).pack(fill=tk.X, pady=2)
        self.thB = tk.IntVar(value=127); self.otsuB = tk.BooleanVar(value=False)
        ttk.Label(left, text="Umbral B").pack(anchor="w")
        tk.Scale(left, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.thB,
                 bg=THEME['bg_secondary'], troughcolor=THEME['accent']).pack(fill=tk.X)
        ttk.Checkbutton(left, text="Usar Otsu", variable=self.otsuB).pack(anchor="w")
        ttk.Button(left, text="Binarizar B", command=self.binarize_B).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        lf = tk.LabelFrame(left, text="Operaciones Lógicas", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        lf.pack(fill=tk.X, pady=2)
        ttk.Button(lf, text="AND", command=self.op_and).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="OR", command=self.op_or).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="XOR", command=self.op_xor).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="NOT A", command=self.op_not_a).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(lf, text="NOT B", command=self.op_not_b).pack(fill=tk.X, padx=4, pady=2)

        rf = tk.LabelFrame(left, text="Operaciones Relacionales", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        rf.pack(fill=tk.X, pady=2)
        ttk.Button(rf, text="A > B", command=self.op_gt).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(rf, text="A < B", command=self.op_lt).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(rf, text="A == B", command=self.op_eq).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(rf, text="A != B", command=self.op_neq).pack(fill=tk.X, padx=4, pady=2)

        grid = tk.Frame(right, bg=THEME['bg_main'])
        grid.pack(expand=True, fill=tk.BOTH)
        self.lblA = tk.Label(grid, text="A", bg=THEME['bg_accent'], fg=THEME['text_primary'], width=50, height=22)
        self.lblB = tk.Label(grid, text="B", bg=THEME['bg_accent'], fg=THEME['text_primary'], width=50, height=22)
        self.lblR = tk.Label(grid, text="Resultado", bg=THEME['bg_accent'], fg=THEME['text_primary'], width=50, height=22)
        self.lblA.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.lblB.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        self.lblR.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="nsew")
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_rowconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)
        self.tkA = self.tkB = self.tkR = None

    # ---- acciones
    def load_A(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.imgA = cv2.imread(path); self.binA = None
        self.tkA = cv_to_tk(self.imgA)
        if self.tkA: self.lblA.configure(image=self.tkA, text="")

    def load_B(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.imgB = cv2.imread(path); self.binB = None
        self.tkB = cv_to_tk(self.imgB)
        if self.tkB: self.lblB.configure(image=self.tkB, text="")

    def binarize_A(self):
        if self.imgA is None: return messagebox.showinfo("Atención", "Carga A primero.")
        self.binA = p2.to_binary(self.imgA, self.thA.get(), self.otsuA.get(), invert=False)
        self.tkA = cv_to_tk(self.binA)
        if self.tkA: self.lblA.configure(image=self.tkA, text="")

    def binarize_B(self):
        if self.imgB is None: return messagebox.showinfo("Atención", "Carga B primero.")
        self.binB = p2.to_binary(self.imgB, self.thB.get(), self.otsuB.get(), invert=False)
        self.tkB = cv_to_tk(self.binB)
        if self.tkB: self.lblB.configure(image=self.tkB, text="")

    def _pair(self):
        if self.binA is None or self.binB is None:
            messagebox.showinfo("Atención", "Binariza A y B primero.")
            return None, None
        return p2.ensure_same_size(self.binA, self.binB)

    def _show_and_save(self, img, name):
        self.result = img
        self.tkR = cv_to_tk(img)
        if self.tkR: self.lblR.configure(image=self.tkR, text="")
        cv2.imwrite(os.path.join("out", f"{name}.png"), img)
        messagebox.showinfo("Guardado", f"Resultado: out/{name}.png")

    # lógicas
    def op_and(self):
        A,B = self._pair()
        if A is None: return
        self._show_and_save(p2.op_and(A,B), "AND")

    def op_or(self):
        A,B = self._pair()
        if A is None: return
        self._show_and_save(p2.op_or(A,B), "OR")

    def op_xor(self):
        A,B = self._pair()
        if A is None: return
        self._show_and_save(p2.op_xor(A,B), "XOR")

    def op_not_a(self):
        if self.binA is None: return messagebox.showinfo("Atención", "Binariza A primero.")
        self._show_and_save(p2.op_not(self.binA), "NOT_A")

    def op_not_b(self):
        if self.binB is None: return messagebox.showinfo("Atención", "Binariza B primero.")
        self._show_and_save(p2.op_not(self.binB), "NOT_B")

    def op_gt(self):
        A,B = self._pair()
        if A is None: return
        self._show_and_save(p2.op_gt(A,B), "A_GT_B")

    def op_lt(self):
        A,B = self._pair()
        if A is None: return
        self._show_and_save(p2.op_lt(A,B), "A_LT_B")

    def op_eq(self):
        A,B = self._pair()
        if A is None: return
        self._show_and_save(p2.op_eq(A,B), "A_EQ_B")

    def op_neq(self):
        A,B = self._pair()
        if A is None: return
        self._show_and_save(p2.op_neq(A,B), "A_NEQ_B")


# ========= Pestaña Práctica 3 =========
class TabPractica3(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=THEME['bg_main'])
        self.img = None
        self.binimg = None
        self.labels4 = self.labels8 = None
        self.color4 = self.color8 = None
        self._build()

    def _build(self):
        # Lado izquierdo con scroll
        sf = ScrollableFrame(self, bg=THEME['bg_secondary'])
        sf.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=8)
        left = sf.inner

        right = tk.Frame(self, bg=THEME['bg_main'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=8, pady=8)

        ttk.Style().theme_use("clam")
        ttk.Label(left, text="Flujo: Cargar → Binarizar → Etiquetar → Mostrar",
                  foreground=THEME['text_secondary'], background=THEME['bg_secondary']).pack(anchor="w", pady=(4,8))
        ttk.Button(left, text="Cargar imagen…", command=self.load_image).pack(fill=tk.X, pady=2)

        binf = tk.LabelFrame(left, text="Binarización", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        binf.pack(fill=tk.X, pady=6)
        self.var_otsu = tk.BooleanVar(value=True)
        self.var_invert = tk.BooleanVar(value=False)
        self.th = tk.IntVar(value=127)
        ttk.Checkbutton(binf, text="Otsu (auto)", variable=self.var_otsu).pack(anchor="w", padx=6)
        ttk.Checkbutton(binf, text="Invertir (fondo↔objeto)", variable=self.var_invert).pack(anchor="w", padx=6)
        tk.Scale(binf, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.th,
                 bg=THEME['bg_secondary'], troughcolor=THEME['accent']).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(binf, text="Aplicar binarización", command=self.apply_binary).pack(fill=tk.X, padx=6, pady=4)

        labf = tk.LabelFrame(left, text="Etiquetado", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        labf.pack(fill=tk.X, pady=6)
        ttk.Button(labf, text="Etiquetar (4 y 8)", command=self.apply_labeling).pack(fill=tk.X, padx=6, pady=4)

        showf = tk.LabelFrame(left, text="Mostrar", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        showf.pack(fill=tk.X, pady=6)
        ttk.Button(showf, text="Original", command=self.show_original).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(showf, text="Binaria", command=self.show_binary).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(showf, text="4-Conex", command=self.show_labels4).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(showf, text="8-Conex", command=self.show_labels8).pack(fill=tk.X, padx=6, pady=2)

        self.viewer = tk.Label(right, text="Aquí se muestra la imagen", bg=THEME['bg_accent'],
                               fg=THEME['text_primary'], width=60, height=28)
        self.viewer.pack(expand=True, fill=tk.BOTH)
        self.tkimg = None

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return messagebox.showerror("Error", "No se pudo cargar la imagen.")
        self.img = img; self.binimg = None
        self.labels4 = self.labels8 = self.color4 = self.color8 = None
        self._show(self.img)

    def apply_binary(self):
        if self.img is None: return messagebox.showinfo("Atención", "Carga una imagen.")
        self.binimg = p3.to_binary(self.img, thresh=self.th.get(),
                                   use_otsu=self.var_otsu.get(), invert=self.var_invert.get())
        self._show(self.binimg)

    def apply_labeling(self):
        if self.binimg is None: return messagebox.showinfo("Atención", "Binariza primero.")
        self.labels4, num4 = p3.label_components(self.binimg, connectivity=4)
        self.labels8, num8 = p3.label_components(self.binimg, connectivity=8)
        self.color4 = p3.colorize_labels(self.labels4)
        self.color8 = p3.colorize_labels(self.labels8)
        print("===== RESULTADOS DE ETIQUETADO =====")
        print(f"Objetos (vecindad 4): {num4}")
        print(f"Objetos (vecindad 8): {num8}")
        print("====================================")

    def _show(self, img):
        self.tkimg = cv_to_tk(img)
        if self.tkimg:
            self.viewer.configure(image=self.tkimg, text="")
        else:
            self.viewer.configure(image="", text="(sin imagen)")

    def show_original(self):
        if self.img is None: return messagebox.showinfo("Info", "No hay imagen cargada.")
        self._show(self.img)

    def show_binary(self):
        if self.binimg is None: return messagebox.showinfo("Info", "Aún no has binarizado.")
        self._show(self.binimg)

    def show_labels4(self):
        if self.color4 is None: return messagebox.showinfo("Info", "Aún no has etiquetado.")
        self._show(self.color4)

    def show_labels8(self):
        if self.color8 is None: return messagebox.showinfo("Info", "Aún no has etiquetado.")
        self._show(self.color8)

# PRACTICA 4
class TabPractica4(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=THEME['bg_main'])
        self.gray = None
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=THEME['bg_secondary'])
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        btn_select = ttk.Button(top, text="Seleccionar imagen…", command=self.select_image)
        btn_select.pack(side=tk.LEFT)

        self.cmap_var = tk.StringVar(self)
        self.cmap_var.set(p4.GRAYSCALE_OPTION)
        self.menu = ttk.OptionMenu(top, self.cmap_var, self.cmap_var.get(), *p4.get_menu_items(), command=lambda *_: self.update_plot())
        self.menu.config(width=32)
        self.menu.pack(side=tk.LEFT, padx=8)

        self.lbl_path = tk.Label(top, text="Sin imagen seleccionada", anchor="w",
                                 bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        self.lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Figura embebida
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Barra inferior
        bottom = tk.Frame(self, bg=THEME['bg_secondary'])
        bottom.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        btn_apply = ttk.Button(bottom, text="Aplicar mapa", command=self.update_plot)
        btn_apply.pack(side=tk.RIGHT, padx=4)

        btn_random = ttk.Button(bottom, text="Nuevo aleatorio", command=self.new_random)
        btn_random.pack(side=tk.RIGHT, padx=4)

    def _refresh_menu_items(self):
        menu = self.menu["menu"]
        menu.delete(0, "end")
        for item in p4.get_menu_items():
            menu.add_command(label=item, command=lambda v=item: (self.cmap_var.set(v), self.update_plot()))

    def new_random(self):
        p4.regenerate_random()
        if self.cmap_var.get() == "Random (custom)":
            self.update_plot()
        self._refresh_menu_items()

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        gray = p4.load_gray(path)
        if gray is None:
            self.lbl_path.config(text="No se pudo cargar la imagen.")
            return
        self.gray = gray
        self.lbl_path.config(text=path)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.axis("off")

        if self.gray is None:
            self.ax.text(0.5, 0.5, "Selecciona una imagen para visualizar",
                         ha="center", va="center", fontsize=12)
            self.canvas.draw()
            return

        selected = self.cmap_var.get()
        colored_rgb, title = p4.apply(self.gray, selected)

        if colored_rgb is None and selected == p4.GRAYSCALE_OPTION:
            self.ax.imshow(self.gray, cmap="gray")
        elif colored_rgb is not None:
            self.ax.imshow(colored_rgb)
        else:
            self.ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
            self.canvas.draw()
            return

        self.ax.set_title(title)
        self.fig.tight_layout()
        self.canvas.draw()

# ========= Pestaña Práctica 5 =========
class TabPractica5(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=THEME['bg_main'])
        self.img = None          # imagen cargada (BGR)
        self.gray = None         # gris
        self.binimg = None       # binaria (0/255)
        self.proc = None         # resultado procesado
        self.tk_src = None
        self.tk_dst = None
        os.makedirs("out_morf", exist_ok=True)
        self._build()

    # ---------- UI ----------
    def _build(self):
        # Lado izquierdo con scroll
        sf = ScrollableFrame(self, bg=THEME['bg_secondary'])
        sf.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=8)
        left = sf.inner

        # Lado derecho: visor
        right = tk.Frame(self, bg=THEME['bg_main'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=8, pady=8)

        # ---- Carga e info
        ttk.Style().theme_use("clam")
        ttk.Button(left, text="Cargar imagen…", command=self.load_image).pack(fill=tk.X, pady=4)
        self.lbl_info = tk.Label(left, text="Sin imagen", anchor="w",
                                 bg=THEME['bg_secondary'], fg=THEME['text_secondary'], wraplength=240, justify="left")
        self.lbl_info.pack(fill=tk.X, pady=(0,6))

        # ---- Binarización
        binf = tk.LabelFrame(left, text="Binarización", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        binf.pack(fill=tk.X, pady=6)
        self.var_otsu = tk.BooleanVar(value=True)
        self.var_invert = tk.BooleanVar(value=False)
        self.th = tk.IntVar(value=127)
        ttk.Checkbutton(binf, text="Otsu (auto)", variable=self.var_otsu).pack(anchor="w", padx=6)
        ttk.Checkbutton(binf, text="Invertir (fondo↔objeto)", variable=self.var_invert).pack(anchor="w", padx=6)
        tk.Scale(binf, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.th,
                 bg=THEME['bg_secondary'], troughcolor=THEME['accent']).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(binf, text="Aplicar binarización", command=self.apply_binary).pack(fill=tk.X, padx=6, pady=4)

        # ---- Parámetros morfología
        par = tk.LabelFrame(left, text="Parámetros morfológicos", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        par.pack(fill=tk.X, pady=6)
        row = tk.Frame(par, bg=THEME['bg_secondary']); row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(row, text="Tamaño EE (impar):", bg=THEME['bg_secondary'], fg=THEME['text_secondary']).pack(side=tk.LEFT)
        self.ksize = tk.IntVar(value=3)
        tk.Spinbox(row, from_=1, to=31, increment=2, width=4, textvariable=self.ksize).pack(side=tk.LEFT, padx=6)

        row2 = tk.Frame(par, bg=THEME['bg_secondary']); row2.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(row2, text="Forma EE:", bg=THEME['bg_secondary'], fg=THEME['text_secondary']).pack(side=tk.LEFT)
        self.shape = tk.StringVar(value="rect")
        ttk.OptionMenu(row2, self.shape, self.shape.get(), "rect", "ellipse", "cross").pack(side=tk.LEFT, padx=6)

        row3 = tk.Frame(par, bg=THEME['bg_secondary']); row3.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(row3, text="Iteraciones:", bg=THEME['bg_secondary'], fg=THEME['text_secondary']).pack(side=tk.LEFT)
        self.iters = tk.IntVar(value=1)
        tk.Spinbox(row3, from_=1, to=10, width=4, textvariable=self.iters).pack(side=tk.LEFT, padx=6)

        # ---- Operaciones básicas
        basic = tk.LabelFrame(left, text="Básicas (gris/binario)", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        basic.pack(fill=tk.X, pady=6)
        ttk.Button(basic, text="Erosión", command=self.op_erosion).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(basic, text="Dilatación", command=self.op_dilatacion).pack(fill=tk.X, padx=6, pady=2)

        # ---- Apertura / Cierre
        ac = tk.LabelFrame(left, text="Apertura / Cierre", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        ac.pack(fill=tk.X, pady=6)
        ttk.Button(ac, text="Apertura (tradicional)", command=self.op_apertura_trad).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(ac, text="Cierre (tradicional)", command=self.op_cierre_trad).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(ac, text="Apertura (OpenCV)", command=self.op_apertura_cv).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(ac, text="Cierre (OpenCV)", command=self.op_cierre_cv).pack(fill=tk.X, padx=6, pady=2)

        # ---- Gradientes / Frontera / Hat
        gf = tk.LabelFrame(left, text="Gradiente / Frontera / Hats", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        gf.pack(fill=tk.X, pady=6)
        ttk.Button(gf, text="Gradiente simétrico", command=lambda: self.op_gradiente("sym")).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(gf, text="Gradiente interno", command=lambda: self.op_gradiente("int")).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(gf, text="Gradiente externo", command=lambda: self.op_gradiente("ext")).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(gf, text="Frontera (binario)", command=self.op_frontera).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(gf, text="Top-Hat", command=self.op_tophat).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(gf, text="Black-Hat", command=self.op_blackhat).pack(fill=tk.X, padx=6, pady=2)

        # ---- Suavizado morfológico
        sm = tk.LabelFrame(left, text="Suavizado morfológico", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        sm.pack(fill=tk.X, pady=6)
        self.smooth_mode = tk.StringVar(value="open_close")
        ttk.OptionMenu(sm, self.smooth_mode, self.smooth_mode.get(), "open_close", "close_open").pack(fill=tk.X, padx=6, pady=2)
        rowp = tk.Frame(sm, bg=THEME['bg_secondary']); rowp.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(rowp, text="Pasadas:", bg=THEME['bg_secondary'], fg=THEME['text_secondary']).pack(side=tk.LEFT)
        self.smooth_passes = tk.IntVar(value=1)
        tk.Spinbox(rowp, from_=1, to=10, width=4, textvariable=self.smooth_passes).pack(side=tk.LEFT, padx=6)
        ttk.Button(sm, text="Aplicar suavizado", command=self.op_smooth).pack(fill=tk.X, padx=6, pady=2)

        # ---- Thinning / Skeleton / Hit-or-Miss
        adv = tk.LabelFrame(left, text="Avanzadas (binario)", bg=THEME['bg_secondary'], fg=THEME['text_primary'])
        adv.pack(fill=tk.X, pady=6)
        ttk.Button(adv, text="Adelgazamiento (Zhang–Suen)", command=self.op_thinning).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(adv, text="Esqueleto morfológico", command=self.op_skeleton).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(adv, text="Hit-or-Miss (presets/editar…)", command=self.op_hitmiss_dialog).pack(fill=tk.X, padx=6, pady=2)

        # ---- Guardar resultado
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Guardar resultado…", command=self.save_result).pack(fill=tk.X, padx=6, pady=2)

        # ---- Visores
        grid = tk.Frame(right, bg=THEME['bg_main'])
        grid.pack(expand=True, fill=tk.BOTH)
        self.lbl_src = tk.Label(grid, text="Original", bg=THEME['bg_accent'], fg=THEME['text_primary'], width=50, height=22)
        self.lbl_dst = tk.Label(grid, text="Procesada", bg=THEME['bg_accent'], fg=THEME['text_primary'], width=50, height=22)
        self.lbl_src.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.lbl_dst.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_rowconfigure(0, weight=1)

    # ---------- Helpers ----------
    def _show_src(self):
        self.tk_src = cv_to_tk(self.img)
        if self.tk_src:
            self.lbl_src.configure(image=self.tk_src, text="")
        else:
            self.lbl_src.configure(image="", text="(sin imagen)")

    def _show_dst(self):
        self.tk_dst = cv_to_tk(self.proc)
        if self.tk_dst:
            self.lbl_dst.configure(image=self.tk_dst, text="")
        else:
            self.lbl_dst.configure(image="", text="(sin resultado)")

    def _ensure_img(self):
        if self.img is None:
            messagebox.showinfo("Atención", "Carga una imagen primero.")
            return False
        return True

    def _kshape(self):
        return int(self.ksize.get()), self.shape.get(), int(self.iters.get())

    # ---------- Acciones ----------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return messagebox.showerror("Error", "No se pudo cargar la imagen.")
        self.img = img
        self.gray = p5._to_gray(img)
        self.binimg = None
        self.proc = None
        h, w = self.gray.shape[:2]
        self.lbl_info.config(text=f"{path}\n{w}×{h}")
        self._show_src()
        self.lbl_dst.configure(image="", text="Procesada")

    def apply_binary(self):
        if not self._ensure_img(): return
        self.binimg = p5.to_binary(self.img, thresh=self.th.get(),
                                   use_otsu=self.var_otsu.get(), invert=self.var_invert.get())
        self.proc = self.binimg.copy()
        self._show_dst()

    # ----- Básicas
    def op_erosion(self):
        if not self._ensure_img(): return
        k,sh,it = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.erode(src, ksize=k, shape=sh, iterations=it)
        self._show_dst()

    def op_dilatacion(self):
        if not self._ensure_img(): return
        k,sh,it = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.dilate(src, ksize=k, shape=sh, iterations=it)
        self._show_dst()

    # ----- Apertura / Cierre
    def op_apertura_trad(self):
        if not self._ensure_img(): return
        k,sh,it = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.apertura_tradicional(src, ksize=k, shape=sh, iterations=it)
        self._show_dst()

    def op_cierre_trad(self):
        if not self._ensure_img(): return
        k,sh,it = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.cierre_tradicional(src, ksize=k, shape=sh, iterations=it)
        self._show_dst()

    def op_apertura_cv(self):
        if not self._ensure_img(): return
        k,sh,it = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.open_morph(src, ksize=k, shape=sh, iterations=it)
        self._show_dst()

    def op_cierre_cv(self):
        if not self._ensure_img(): return
        k,sh,it = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.close_morph(src, ksize=k, shape=sh, iterations=it)
        self._show_dst()

    # ----- Gradientes / Frontera / Hats
    def op_gradiente(self, modo):
        if not self._ensure_img(): return
        k,sh,_ = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.gradient(src, ksize=k, shape=sh, mode=modo)
        self._show_dst()

    def op_frontera(self):
        if not self._ensure_img(): return
        if self.binimg is None:
            messagebox.showinfo("Atención", "Frontera requiere imagen binaria.\nAplica binarización primero.")
            return
        k,sh,_ = self._kshape()
        self.proc = p5.boundary(self.binimg, ksize=k, shape=sh)
        self._show_dst()

    def op_tophat(self):
        if not self._ensure_img(): return
        k,sh,_ = self._kshape()
        src = self.gray
        self.proc = p5.top_hat(src, ksize=k, shape=sh)
        self._show_dst()

    def op_blackhat(self):
        if not self._ensure_img(): return
        k,sh,_ = self._kshape()
        src = self.gray
        self.proc = p5.black_hat(src, ksize=k, shape=sh)
        self._show_dst()

    # ----- Suavizado
    def op_smooth(self):
        if not self._ensure_img(): return
        k,sh,_ = self._kshape()
        src = self.binimg if self.binimg is not None else self.gray
        self.proc = p5.smooth(src, ksize=k, shape=sh, passes=self.smooth_passes.get(), mode=self.smooth_mode.get())
        self._show_dst()

    # ----- Avanzadas (binario)
    def op_thinning(self):
        if not self._ensure_img(): return
        if self.binimg is None:
            messagebox.showinfo("Atención", "Adelgazamiento requiere imagen binaria.\nAplica binarización primero.")
            return
        self.proc = p5.thinning(self.binimg)
        self._show_dst()

    def op_skeleton(self):
        if not self._ensure_img(): return
        if self.binimg is None:
            messagebox.showinfo("Atención", "Esqueleto requiere imagen binaria.\nAplica binarización primero.")
            return
        k,sh,_ = self._kshape()
        self.proc = p5.skeletonize(self.binimg, ksize=k, shape=sh)
        self._show_dst()

    def op_hitmiss_dialog(self):
        if not self._ensure_img(): return
        if self.binimg is None:
            messagebox.showinfo("Atención", "Hit-or-Miss requiere imagen binaria.\nAplica binarización primero.")
            return

        # Ventana simple para editar un kernel 3x3 (0/1) con presets
        dialog = tk.Toplevel(self)
        dialog.title("Hit-or-Miss - Kernel 3×3")
        dialog.configure(bg=THEME['bg_secondary'])
        vals = [[tk.IntVar(value=0) for _ in range(3)] for _ in range(3)]

        presets = {
            "Cruz":       [[0,1,0],[1,1,1],[0,1,0]],
            "Cuadro":     [[1,1,1],[1,1,1],[1,1,1]],
            "Esquina UL": [[1,1,0],[1,1,0],[0,0,0]],
            "T":          [[1,1,1],[0,1,0],[0,1,0]],
        }

        def load_preset(name):
            k = presets[name]
            for i in range(3):
                for j in range(3):
                    vals[i][j].set(k[i][j])

        # Grid de checks
        grid = tk.Frame(dialog, bg=THEME['bg_secondary'])
        grid.pack(padx=8, pady=8)
        for i in range(3):
            for j in range(3):
                tk.Checkbutton(grid, variable=vals[i][j], bg=THEME['bg_secondary']).grid(row=i, column=j, padx=4, pady=4)

        # Presets
        presetf = tk.Frame(dialog, bg=THEME['bg_secondary']); presetf.pack(padx=8, pady=(0,8), fill=tk.X)
        ttk.Label(presetf, text="Preset:").pack(side=tk.LEFT)
        pvar = tk.StringVar(value="Cruz")
        menu = ttk.OptionMenu(presetf, pvar, pvar.get(), *presets.keys(), command=lambda n: load_preset(n))
        menu.pack(side=tk.LEFT, padx=6)
        load_preset("Cruz")

        def apply_hitmiss():
            kh = np.array([[vals[i][j].get() for j in range(3)] for i in range(3)], dtype=np.uint8)
            try:
                self.proc = p5.hit_or_miss(self.binimg, kernel_hit=kh)
                self._show_dst()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(dialog, text="Aplicar", command=apply_hitmiss).pack(pady=(0,8))
        dialog.transient(self)
        dialog.grab_set()
        dialog.wait_window()

    def save_result(self):
        if self.proc is None:
            return messagebox.showinfo("Info", "No hay resultado para guardar.")
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialdir="out_morf",
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tif;*.tiff")])
        if not path: return
        try:
            cv2.imwrite(path, self.proc)
            messagebox.showinfo("Guardado", f"Resultado guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar.\n{e}")

# ========= App raíz (Notebook) =========
def run_app():
    root = tk.Tk()
    root.title("Procesamiento de Imágenes - Interfaz Unificada")
    root.geometry("1280x720")
    root.configure(bg=THEME['bg_main'])

    nb = ttk.Notebook(root)
    nb.pack(expand=True, fill=tk.BOTH)

    nb.add(TabPractica1(nb), text="Práctica 1: Básico & Histogramas")
    nb.add(TabPractica2(nb), text="Práctica 2: Lógicas & Relacionales")
    nb.add(TabPractica3(nb), text="Práctica 3: Etiquetado 4 vs 8")
    nb.add(TabPractica4(nb), text="Práctica 4: Pseudocolor")
    nb.add(TabPractica5(nb), text="Práctica 5: Morfología")

    root.mainloop()

# Ejecutar directamente si se llama como script
if __name__ == "__main__":
    run_app()
