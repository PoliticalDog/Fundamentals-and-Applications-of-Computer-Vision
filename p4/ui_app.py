# ui_app.py
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

# Matplotlib (solo render embebido)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------- Importa TODA la lógica del módulo core ---------
from img_core import (
    # visualización / conversiones
    mosaic_rgb_channels, to_gray_if_needed, mosaic_cmyk, mosaic_hsl,
    # histogramas / stats
    compute_hist_rgb, compute_hist_gray, compute_stats_rgb,
    # binarización
    to_binary,
    # CC
    ensure_same_size, label_components, colorize_labels,
    # lógicas/relacionales
    logic_and, logic_or, logic_xor, logic_not,
    rel_gt, rel_lt, rel_eq, rel_neq
)

# ========= Tema / utilidades de UI =========
NORD = {
    'bg_main': '#2E3440','bg_secondary': '#3B4252','bg_accent': '#4C566A',
    'text_primary': '#ECEFF4','text_secondary': '#D8DEE9','button_primary': '#5E81AC',
    'button_success': '#A3BE8C','button_warning': '#EBCB8B','button_danger': '#BF616A',
    'button_info': '#88C0D0','button_accent': '#81A1C1','accent': '#81A1C1','border': '#434C5E'
}

def apply_style(root):
    root.configure(bg=NORD['bg_main'])
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=NORD['bg_main'])
    style.configure("TLabel", background=NORD['bg_main'], foreground=NORD['text_primary'])
    style.configure("TLabelframe", background=NORD['bg_secondary'], foreground=NORD['text_primary'])
    style.configure("TButton", padding=6, font=("Segoe UI", 9, "bold"))
    style.configure("Accent.TButton", background=NORD['button_primary'], foreground="white")
    style.map("Accent.TButton", background=[("active", NORD['button_info'])])

def cv_to_tk(img, max_size=(420, 420)):
    """SOLO para UI: convierte ndarray (BGR o GRAY) → PhotoImage escalada."""
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

# ========= Ventana (UI pura) =========
class OneWindowApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Procesamiento de Imágenes — UI (lógica en img_core.py)")
        self.geometry("1380x860")
        apply_style(self)

        # Carpeta "Imagenes" al mismo nivel que este archivo
        self.images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Imagenes")
        os.makedirs(self.images_dir, exist_ok=True)

        # Estado
        self.imgA = None
        self.imgB = None
        self.binA = None
        self.binB = None
        self.result = None

        # ---- Layout base
        left = tk.Frame(self, bg=NORD['bg_secondary'], bd=2, relief="groove", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right = tk.Frame(self, bg=NORD['bg_main'])
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # ====== Panel izquierdo (controles)
        ttk.Label(left, text="1) Cargar imágenes", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4,2))
        ttk.Button(left, text="Cargar Imagen 1", command=self.load_A, style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Cargar Imagen 2", command=self.load_B, style="Accent.TButton").pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="2) ¿A qué imagen aplicar?", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.target = tk.StringVar(value="A")
        ttk.Radiobutton(left, text="Imagen 1", variable=self.target, value="A").pack(anchor="w")
        ttk.Radiobutton(left, text="Imagen 2", variable=self.target, value="B").pack(anchor="w")
        ttk.Radiobutton(left, text="Ambas (si aplica)", variable=self.target, value="Ambas").pack(anchor="w")

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="3) Selecciona transformación", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.op_group = tk.StringVar(value="Básicos")
        self.op_name  = tk.StringVar(value="Convertir a Grises")

        groups = ["Básicos", "Binarización", "Histogramas/Stats", "Etiquetado CC", "Lógicas/Relacionales"]
        self.cmb_group = ttk.Combobox(left, values=groups, textvariable=self.op_group, state="readonly")
        self.cmb_group.pack(fill=tk.X, pady=2)
        self.cmb_group.bind("<<ComboboxSelected>>", self._refresh_ops)

        self.cmb_ops = ttk.Combobox(left, values=[], textvariable=self.op_name, state="readonly")
        self.cmb_ops.pack(fill=tk.X, pady=2)

        # Parámetros
        spec = tk.LabelFrame(left, text="Parámetros", bg=NORD['bg_secondary'], fg=NORD['text_primary'])
        spec.pack(fill=tk.X, pady=8)

        # Binarización
        self.var_otsu = tk.BooleanVar(value=False)
        self.var_invert = tk.BooleanVar(value=False)
        self.th = tk.IntVar(value=127)

        ttk.Label(spec, text="Umbral").pack(anchor="w")
        self.scl_th = tk.Scale(spec, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.th,
                               bg=NORD['bg_secondary'], troughcolor=NORD['bg_accent'], highlightthickness=0)
        self.scl_th.pack(fill=tk.X)
        ttk.Checkbutton(spec, text="Usar Otsu", variable=self.var_otsu).pack(anchor="w")
        ttk.Checkbutton(spec, text="Invertir (fondo↔objeto)", variable=self.var_invert).pack(anchor="w")

        # Histogramas
        ttk.Label(spec, text="Canal (histogramas)").pack(anchor="w", pady=(6,0))
        self.cmb_canal = ttk.Combobox(spec, values=["Todos", "Rojo", "Verde", "Azul", "Grises"], state="readonly")
        self.cmb_canal.set("Todos")
        self.cmb_canal.pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Button(left, text="Aplicar", command=self.apply, style="Accent.TButton").pack(fill=tk.X)

        # Botón clásico (dejarlo por si lo usas)
        ttk.Button(left, text="Guardar como... (elige ruta)", command=self.save_result_dialog)\
            .pack(fill=tk.X, pady=(6,2))

        # NUEVO: Guardar directo en carpeta "Imagenes" como PNG y JPG
        ttk.Button(left, text="Guardar en 'Imagenes' (PNG + JPG)", command=self.save_to_images_folder)\
            .pack(fill=tk.X, pady=(2,0))

        self.lbl_status = ttk.Label(left, text="Listo.")
        self.lbl_status.pack(anchor="w", pady=(8,0))

        # ====== Panel derecho: PanedWindow (redimensionable)
        paned = ttk.Panedwindow(right, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Arriba: visores
        views_frame = tk.Frame(paned, bg=NORD['bg_main'])
        self.lblA = tk.Label(views_frame, text="Imagen 1", bg=NORD['bg_accent'], fg=NORD['text_primary'], width=48, height=20)
        self.lblB = tk.Label(views_frame, text="Imagen 2", bg=NORD['bg_accent'], fg=NORD['text_primary'], width=48, height=20)
        self.lblR = tk.Label(views_frame, text="Resultado", bg=NORD['bg_accent'], fg=NORD['text_primary'], width=48, height=20)

        self.lblA.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.lblB.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        self.lblR.grid(row=0, column=2, padx=6, pady=6, sticky="nsew")

        for c in range(3):
            views_frame.grid_columnconfigure(c, weight=1)
        views_frame.grid_rowconfigure(0, weight=1)

        # Abajo: análisis (histograma + stats)
        analysis_frame = tk.LabelFrame(paned, text="Análisis (histograma + stats)",
                                       bg=NORD['bg_secondary'], fg=NORD['text_primary'])
        analysis_frame.pack_propagate(False)

        # Figura compacta
        self.fig = Figure(figsize=(6.4, 2.2), dpi=100, facecolor=NORD['bg_main'])
        self.ax = self.fig.add_subplot(111, facecolor=NORD['bg_secondary'])
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=analysis_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6,3), pady=6)

        # Stats
        self.text_stats = tk.Text(analysis_frame, height=12, width=40,
                                  bg=NORD['bg_secondary'], fg=NORD['text_primary'],
                                  insertbackground=NORD['text_primary'],
                                  selectbackground=NORD['button_accent'],
                                  selectforeground=NORD['text_primary'],
                                  font=('Consolas', 9), relief='flat', bd=0)
        self.text_stats.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(3,6), pady=6)

        paned.add(views_frame, weight=3)
        paned.add(analysis_frame, weight=1)
        self.after(50, lambda: paned.sashpos(0, int(self.winfo_height()*0.65)))

        # Inicializar combos
        self._refresh_ops()

    # ====== Helpers UI ======
    def _style_axes(self):
        self.ax.tick_params(colors=NORD['text_primary'])
        self.ax.xaxis.label.set_color(NORD['text_primary'])
        self.ax.yaxis.label.set_color(NORD['text_primary'])
        self.ax.title.set_color(NORD['text_primary'])
        for s in ['bottom','top','right','left']:
            self.ax.spines[s].set_color(NORD['text_secondary'])

    def _refresh_ops(self, *_):
        group = self.op_group.get()
        ops = []
        if group == "Básicos":
            ops = ["Mostrar RGB por canales", "Convertir a Grises", "Convertir a CMYK", "Convertir a HSL"]
            self._toggle_params(show_th=False, show_otsu=False, show_invert=False, show_canal=False)
        elif group == "Binarización":
            ops = ["Binarizar (umbral)", "Binarizar (Otsu)"]
            self._toggle_params(show_th=True, show_otsu=True, show_invert=True, show_canal=False)
        elif group == "Histogramas/Stats":
            ops = ["Histograma", "Stats RGB (embebido)"]
            self._toggle_params(show_th=False, show_otsu=False, show_invert=False, show_canal=True)
        elif group == "Etiquetado CC":
            ops = ["Etiquetar 4/8 y colorear"]
            self._toggle_params(show_th=True, show_otsu=True, show_invert=True, show_canal=False)
        elif group == "Lógicas/Relacionales":
            ops = ["AND", "OR", "XOR", "NOT A", "NOT B", "A > B", "A < B", "A == B", "A != B"]
            self._toggle_params(show_th=False, show_otsu=False, show_invert=False, show_canal=False)
        self.cmb_ops['values'] = ops
        if ops:
            self.cmb_ops.set(ops[0])

    def _toggle_params(self, show_th, show_otsu, show_invert, show_canal):
        self.scl_th.configure(state="normal" if show_th else "disabled")
        self.cmb_canal.configure(state="readonly" if show_canal else "disabled")
        self._show_params_show = (show_th, show_otsu, show_invert, show_canal)

    def _update_view(self, which, img):
        tkimg = cv_to_tk(img, max_size=(520, 520))
        if which == "A":
            self.lblA.configure(image=tkimg, text=""); self.lblA.image = tkimg
        elif which == "B":
            self.lblB.configure(image=tkimg, text=""); self.lblB.image = tkimg
        elif which == "R":
            self.lblR.configure(image=tkimg, text=""); self.lblR.image = tkimg

    # ====== Carga ======
    def load_A(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path: return
        self.imgA = cv2.imread(path, cv2.IMREAD_COLOR)
        self.binA = None
        self._update_view("A", self.imgA)

    def load_B(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path: return
        self.imgB = cv2.imread(path, cv2.IMREAD_COLOR)
        self.binB = None
        self._update_view("B", self.imgB)

    # ====== Aplicar (UI -> llama a lógica en img_core) ======
    def apply(self):
        group = self.op_group.get()
        op = self.cmb_ops.get()
        tgt = self.target.get()
        show_th, show_otsu, show_invert, show_canal = self._show_params_show

        thresh = self.th.get() if show_th else None
        use_otsu = self.var_otsu.get() if show_otsu else False
        invert = self.var_invert.get() if show_invert else False
        canal = self.cmb_canal.get() if show_canal else None

        try:
            if group == "Básicos":
                self._apply_basic(op, tgt)
            elif group == "Binarización":
                self._apply_binarization(op, tgt, thresh, use_otsu, invert)
            elif group == "Histogramas/Stats":
                self._apply_hist_stats(op, tgt, canal)
            elif group == "Etiquetado CC":
                self._apply_labeling(tgt, thresh, use_otsu, invert)
            elif group == "Lógicas/Relacionales":
                self._apply_logic(op)
            self.lbl_status.config(text=f"OK: {group} → {op} ({tgt})")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.lbl_status.config(text="Error (ver mensaje).")

    # ---- Básicos
    def _apply_basic(self, op, tgt):
        def do_for(img):
            if img is None: return None
            if op == "Mostrar RGB por canales": return mosaic_rgb_channels(img)
            if op == "Convertir a Grises":      return to_gray_if_needed(img)
            if op == "Convertir a CMYK":        return mosaic_cmyk(img)
            if op == "Convertir a HSL":         return mosaic_hsl(img)
            return None

        if tgt == "A":   res = do_for(self.imgA)
        elif tgt == "B": res = do_for(self.imgB)
        else:
            resA = do_for(self.imgA); resB = do_for(self.imgB)
            if resA is None and resB is None: return
            if resA is None: res = resB
            elif resB is None: res = resA
            else:
                h = max(resA.shape[0], resB.shape[0])
                def pad(im): 
                    if im.shape[0] == h: return im
                    return cv2.copyMakeBorder(im, 0, h-im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                res = np.hstack([pad(resA), pad(resB)])
        if res is not None:
            self.result = res
            self._update_view("R", res)

    # ---- Binarización
    def _apply_binarization(self, op, tgt, thresh, use_otsu, invert):
        def do_for(img):
            if img is None: return None
            if op == "Binarizar (Otsu)":
                return to_binary(img, use_otsu=True, invert=invert)
            else:
                return to_binary(img, thresh=thresh if thresh is not None else 127,
                                 use_otsu=use_otsu, invert=invert)

        if tgt == "A":   self.binA = do_for(self.imgA); res = self.binA
        elif tgt == "B": self.binB = do_for(self.imgB); res = self.binB
        else:
            resA = do_for(self.imgA); resB = do_for(self.imgB)
            if resA is not None: self.binA = resA
            if resB is not None: self.binB = resB
            if resA is None and resB is None: return
            if resA is None: res = resB
            elif resB is None: res = resA
            else:
                A, B = ensure_same_size(resA, resB)
                res = np.hstack([A, B])
        if res is not None:
            self.result = res
            self._update_view("R", res)

    # ---- Histogramas/Stats
    def _apply_hist_stats(self, op, tgt, canal):
        img = self.imgA if tgt == "A" else self.imgB if tgt == "B" else None
        if img is None:
            messagebox.showinfo("Atención", "Selecciona Imagen 1 o Imagen 2 (no 'Ambas') y cárgala.")
            return

        self.ax.clear(); self._style_axes()
        self.ax.set_facecolor(NORD['bg_secondary'])
        self.ax.grid(True, color=NORD['text_secondary'], alpha=0.3)

        if op == "Histograma":
            if canal == "Grises":
                hist = compute_hist_gray(img)
                self.ax.plot(hist, color='gray')
                self.ax.set_title("Histograma - Grises")
            else:
                hists = compute_hist_rgb(img)
                if hists is None: return
                if canal == "Todos":
                    self.ax.plot(hists['Rojo'],  label='Rojo',  color='red')
                    self.ax.plot(hists['Verde'], label='Verde', color='green')
                    self.ax.plot(hists['Azul'],  label='Azul',  color='blue')
                    self.ax.legend()
                    self.ax.set_title("Histograma - Todos")
                else:
                    color_map = {'Rojo':'red','Verde':'green','Azul':'blue'}
                    self.ax.plot(hists[canal], label=canal, color=color_map[canal])
                    self.ax.legend(); self.ax.set_title(f"Histograma - {canal}")
            self.ax.set_xlabel("Intensidad"); self.ax.set_ylabel("Frecuencia")
            self.fig.tight_layout()
            self.canvas.draw_idle()

        elif op == "Stats RGB (embebido)":
            stats = compute_stats_rgb(img)
            if stats is None: return
            self.text_stats.delete("1.0", tk.END)
            for ckey, props in stats.items():
                self.text_stats.insert(tk.END, f"Canal {ckey}:\n")
                for prop, val in props.items():
                    self.text_stats.insert(tk.END, f"  {prop}: {val:.4f}\n")
                self.text_stats.insert(tk.END, "\n")

    # ---- Etiquetado
    def _apply_labeling(self, tgt, thresh, use_otsu, invert):
        img = self.imgA if tgt == "A" else self.imgB if tgt == "B" else None
        if img is None:
            messagebox.showinfo("Atención", "Selecciona Imagen 1 o Imagen 2 (no 'Ambas') y cárgala.")
            return
        binimg = to_binary(img, thresh=thresh if thresh is not None else 127,
                           use_otsu=use_otsu, invert=invert)
        labels4, num4 = label_components(binimg, connectivity=4)
        labels8, num8 = label_components(binimg, connectivity=8)
        color4 = colorize_labels(labels4)
        color8 = colorize_labels(labels8)
        A, B = ensure_same_size(color4, color8)
        res = np.hstack([A, B])
        self.result = res
        self._update_view("R", res)
        self.lbl_status.config(text=f"Etiquetas → 4: {num4} | 8: {num8}")

    # ---- Lógicas/Relacionales
    def _apply_logic(self, op):
        if self.imgA is None or self.imgB is None:
            messagebox.showinfo("Atención", "Carga ambas imágenes para operaciones lógicas/relacionales.")
            return
        if self.binA is None: self.binA = to_binary(self.imgA, use_otsu=True)
        if self.binB is None: self.binB = to_binary(self.imgB, use_otsu=True)

        A, B = ensure_same_size(self.binA, self.binB)
        if   op == "AND":   res = logic_and(A, B)
        elif op == "OR":    res = logic_or(A, B)
        elif op == "XOR":   res = logic_xor(A, B)
        elif op == "NOT A": res = logic_not(A)
        elif op == "NOT B": res = logic_not(B)
        elif op == "A > B": res = rel_gt(A, B)
        elif op == "A < B": res = rel_lt(A, B)
        elif op == "A == B": res = rel_eq(A, B)
        elif op == "A != B": res = rel_neq(A, B)
        else: return
        self.result = res
        self._update_view("R", res)

    # ====== Guardar ======
    def save_result_dialog(self):
        """Opción clásica: escoger ruta/archivo manualmente."""
        if self.result is None:
            messagebox.showinfo("Atención", "No hay resultado para guardar.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png", initialfile="resultado.png",
                                                filetypes=[("PNG", "*.png"), ("JPG", "*.jpg;*.jpeg")])
        if not filename: 
            return
        ok = cv2.imwrite(filename, self.result)
        if ok:
            messagebox.showinfo("Guardado", f"Guardado en:\n{filename}")
        else:
            messagebox.showerror("Error", "No se pudo guardar el archivo.")

    def save_to_images_folder(self):
        """
        NUEVO: guarda automáticamente el resultado en la carpeta 'Imagenes' (al lado de este .py),
        en ambos formatos: PNG y JPG. Usa un nombre base con timestamp para evitar sobrescribir.
        """
        if self.result is None:
            messagebox.showinfo("Atención", "No hay resultado para guardar.")
            return

        base = datetime.now().strftime("resultado_%Y%m%d_%H%M%S")
        png_path = os.path.join(self.images_dir, f"{base}.png")
        jpg_path = os.path.join(self.images_dir, f"{base}.jpg")

        ok_png = cv2.imwrite(png_path, self.result)
        ok_jpg = cv2.imwrite(jpg_path, self.result)

        if ok_png and ok_jpg:
            messagebox.showinfo("Guardado",
                                f"Imágenes guardadas:\n• {png_path}\n• {jpg_path}")
        elif ok_png:
            messagebox.showwarning("Parcial",
                                   f"PNG guardado, pero JPG falló.\nPNG: {png_path}")
        elif ok_jpg:
            messagebox.showwarning("Parcial",
                                   f"JPG guardado, pero PNG falló.\nJPG: {jpg_path}")
        else:
            messagebox.showerror("Error", "No se pudo guardar ni PNG ni JPG.")

if __name__ == "__main__":
    OneWindowApp().mainloop()
