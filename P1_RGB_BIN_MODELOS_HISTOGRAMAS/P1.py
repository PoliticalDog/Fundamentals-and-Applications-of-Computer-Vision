# Practica1VPC_gui.py
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from scipy.stats import skew
from math import log2
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# --------------------------
# FUNCIONES DE PROCESAMIENTO (NO MODIFICADAS, SOLO ADAPTADAS PARA GUI)
# --------------------------
#CARGAR IMAGEN
def cargar_imagen(ruta):
    imagen = Image.open(ruta)
    plt.imshow(imagen)
    plt.title("Imagen original")
    plt.axis("off")
    plt.show()
    return imagen

#SEPARAR RGB en 3 variables y mostrar
def separar_rgb(imagen):
    r, g, b = imagen.split()
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap='Reds')
    plt.title("Componente R")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='Greens')
    plt.title("Componente G")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap='Blues')
    plt.title("Componente B")
    plt.axis("off")

    plt.show()

def convertir_a_grises(imagen):
    imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    plt.imshow(imagen_cv, cmap='gray')
    plt.title("Imagen en escala de grises")
    plt.axis("off")
    plt.show()
    return imagen_cv

def binarizar_imagen(imagen_gris, umbral=128):
    #retval, dst = cv2.threshold(src, thresh, maxval, type)
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    plt.imshow(binaria, cmap='gray')
    plt.title(f"Imagen binarizada (umbral = {umbral})")
    plt.axis("off")
    plt.show()

def convertir_a_cmyk(imagen_pil):
    """Convierte imagen PIL a CMYK y muestra los canales"""
    cmyk = imagen_pil.convert("CMYK")
    c, m, y, k = cmyk.split()
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(c, cmap='gray')
    plt.title("Cyan")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(m, cmap='gray')
    plt.title("Magenta")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(y, cmap='gray')
    plt.title("Yellow")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(k, cmap='gray')
    plt.title("Black")
    plt.axis("off")
    
    plt.show()
    return cmyk

def convertir_a_hsl(imagen_pil):
    """Convierte imagen PIL a HSL y muestra los canales"""
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2HLS)  # OpenCV usa HLS
    h, l, s = cv2.split(imagen_cv)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(h, cmap='hsv')
    plt.title("Hue")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(l, cmap='gray')
    plt.title("Lightness")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(s, cmap='gray')
    plt.title("Saturation")
    plt.axis("off")
    
    plt.show()
    return imagen_cv

def binarizar_imagen_umbral(imagen_gris, umbral):
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    plt.imshow(binaria, cmap='gray')
    plt.title(f"Binarizada (umbral={umbral})")
    plt.axis("off")
    plt.show()
    return binaria

def binarizar_imagen_otsu(imagen_gris):
    _, binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(binaria, cmap='gray')
    plt.title("Binarizada (Método Otsu)")
    plt.axis("off")
    plt.show()
    return binaria

def calcular_histogramas_rgb(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    resultados = {}
    colores = {"Rojo": "red", "Verde": "green", "Azul": "blue"}

    for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
        datos = imagen_rgb[:, :, i].flatten()

        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
        prob = histograma / histograma.sum()

        energia = np.sum(prob ** 2)
        entropia = -np.sum([p * log2(p) for p in prob if p > 0])
        asimetria = skew(datos)
        media = np.mean(datos)
        varianza = np.var(datos)

        resultados[canal] = {
            'Energía': energia,
            'Entropía': entropia,
            'Asimetría': asimetria,
            'Media': media,
            'Varianza': varianza
        }

        plt.figure()
        plt.title(f'Histograma del canal {canal}')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia')
        plt.plot(histograma, color=colores[canal])
        plt.grid(True)
        plt.show()

    for canal, props in resultados.items():
        print(f'\nCanal {canal}:')
        for prop, valor in props.items():
            print(f'  {prop}: {valor:.4f}')

    for canal, props in resultados.items():
        print(f'\nCanal {canal}:')
        for prop, valor in props.items():
            print(f'  {prop}: {valor:.4f}')

def calcular_histograma_grises(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    datos = imagen.flatten()

    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
    prob = histograma / histograma.sum()

    energia = np.sum(prob ** 2)
    entropia = -np.sum([p * log2(p) for p in prob if p > 0])
    asimetria = skew(datos)
    media = np.mean(datos)
    varianza = np.var(datos)

    print("\nPropiedades de la imagen en escala de grises:")
    print(f" Energía: {energia:.4f}")
    print(f" Entropía: {entropia:.4f}")
    print(f" Asimetría: {asimetria:.4f}")
    print(f" Media: {media:.2f}")
    print(f" Varianza: {varianza:.2f}")

    plt.figure()
    plt.title('Histograma de imagen en escala de grises')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    plt.plot(histograma, color='gray')
    plt.grid(True)
    plt.show()

# --------------------------
# FUNCIONES AUXILIARES PARA LA GUI (NO CAMBIAN TRATAMIENTO)
# --------------------------
def compute_histogramas_rgb_arrays(ruta_imagen):
    """Devuelve dict {'Rojo': histR, 'Verde': histG, 'Azul': histB} usando np.histogram (como lógica)"""
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return None
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = {}
    for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
        datos = imagen_rgb[:, :, i].flatten()
        # bins se usa para definir los intervalos del histograma (cajones en lso que cae cada valor)
        #histograma, limites
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
        resultados[canal] = histograma
    return resultados

def compute_histogram_gris_array(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return None
    datos = imagen.flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
    return histograma

# --------------------------
# INTERFAZ GRAFICA
# --------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de Imagen - Interfaz")
        self.ruta_imagen = None
        self.imagen_pil = None
        self.imagen_gris = None
        
        # Configuración de colores del tema
        self.colors = {
            'bg_main': '#2E3440',           # Azul gris oscuro principal
            'bg_secondary': '#3B4252',      # Azul gris más claro para paneles
            'bg_accent': '#4C566A',         # Gris azulado para acentos
            'text_primary': '#ECEFF4',      # Blanco nieve para texto principal
            'text_secondary': '#D8DEE9',    # Gris claro para texto secundario
            'button_primary': '#5E81AC',    # Azul profesional para botones principales
            'button_success': '#A3BE8C',    # Verde para acciones exitosas
            'button_warning': '#EBCB8B',    # Amarillo para advertencias
            'button_danger': '#BF616A',     # Rojo para acciones peligrosas
            'button_info': '#88C0D0',       # Azul claro para información
            'button_accent': '#81A1C1',     # Azul celeste para acentos
            'accent': '#81A1C1',            # Azul celeste para acentos
            'border': '#434C5E'             # Borde sutil
        }
        
        # Configurar el fondo principal de la ventana
        self.root.configure(bg=self.colors['bg_main'])
        
        self.setup_ui()

    def setup_ui(self):
        frame_left = tk.Frame(root, padx=8, pady=8, bg=self.colors['bg_secondary'])
        frame_left.pack(side=tk.LEFT, fill=tk.Y)

        frame_right = tk.Frame(root, padx=8, pady=8, bg=self.colors['bg_main'])
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ------------- Botones
        #Boton cargar imagen
        tk.Button(frame_left, text="Cargar Imagen", width=20, command=self.seleccionar_imagen,
                 bg=self.colors['button_primary'], fg=self.colors['text_primary'], 
                 activebackground=self.colors['button_accent'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10, 'bold'), relief='flat', bd=0).pack(pady=4)
        #btones de operaciones para separar RGB
        tk.Button(frame_left, text="Separar RGB", width=20, command=self.ejecutar_separar_rgb,
                 bg=self.colors['button_info'], fg=self.colors['text_primary'],
                 activebackground=self.colors['button_accent'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=4)
        #boton para ejecutar escala de grises
        tk.Button(frame_left, text="Convertir a Grises", width=20, command=self.ejecutar_grises,
                 bg=self.colors['button_info'], fg=self.colors['text_primary'],
                 activebackground=self.colors['button_accent'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=4)
        #boton para ejecutar binarizacion
        tk.Button(frame_left, text="Binarizar", width=20, command=self.ejecutar_binarizar,
                 bg=self.colors['button_info'], fg=self.colors['text_primary'],
                 activebackground=self.colors['button_accent'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=4)
        # Conversiones extra
        tk.Button(frame_left, text="Convertir a CMYK", width=20, command=self.ejecutar_cmyk,
                 bg=self.colors['button_info'], fg=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=4)

        tk.Button(frame_left, text="Convertir a HSL", width=20, command=self.ejecutar_hsl,
                 bg=self.colors['button_info'], fg=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=4)

        # Binarización extra
        tk.Button(frame_left, text="Binarizar (Otsu)", width=20, command=self.ejecutar_binarizar_otsu,
                 bg=self.colors['button_warning'], fg=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=4)
        
        # botones para mostrar histogramas
        tk.Label(frame_left, text="Histogramas RGB:", anchor="w", 
                bg=self.colors['bg_secondary'], fg=self.colors['text_primary'], 
                font=('Arial', 10, 'bold')).pack(pady=(12,2))
        tk.Button(frame_left, text="Rojo", width=20, command=lambda: self.mostrar_hist("Rojo"),
                 bg='#BF616A', fg=self.colors['text_primary'],
                 activebackground='#D08770', activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=2)
        tk.Button(frame_left, text="Verde", width=20, command=lambda: self.mostrar_hist("Verde"),
                 bg='#A3BE8C', fg=self.colors['text_primary'],
                 activebackground='#8FBCBB', activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=2)
        tk.Button(frame_left, text="Azul", width=20, command=lambda: self.mostrar_hist("Azul"),
                 bg='#5E81AC', fg=self.colors['text_primary'],
                 activebackground='#81A1C1', activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=2)
        tk.Button(frame_left, text="Todos", width=20, command=lambda: self.mostrar_hist("Todos"),
                 bg=self.colors['button_accent'], fg=self.colors['text_primary'],
                 activebackground=self.colors['button_primary'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=2)

        tk.Label(frame_left, text="Histograma Grises:", anchor="w",
                bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                font=('Arial', 10, 'bold')).pack(pady=(12,2))
        tk.Button(frame_left, text="Ver Grises", width=20, command=lambda: self.mostrar_hist("Grises"),
                 bg=self.colors['bg_accent'], fg=self.colors['text_primary'],
                 activebackground=self.colors['button_accent'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=2)

        tk.Label(frame_left, text="Operaciones estadísticas:", anchor="w",
                bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                font=('Arial', 10, 'bold')).pack(pady=(12,2))
        tk.Button(frame_left, text="Calcular", width=20, command=self.calcular_stats_via_funciones,
                 bg=self.colors['button_success'], fg=self.colors['text_primary'],
                 activebackground=self.colors['button_accent'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=2)
        tk.Button(frame_left, text="Calcular RGB", width=20, command=lambda: self.mostrar_stats_embebidos(),
                 bg=self.colors['button_success'], fg=self.colors['text_primary'],
                 activebackground=self.colors['button_accent'], activeforeground=self.colors['text_primary'],
                 font=('Arial', 10), relief='flat', bd=0).pack(pady=2)


        # Panel derecho: imagen y canvas del histograma
        self.panel_imagen = tk.Label(frame_right, bg=self.colors['bg_main'])
        self.panel_imagen.pack(side=tk.TOP, pady=4)

        # Figura matplotlib embebida (histograma)
        self.fig = Figure(figsize=(6,4), facecolor=self.colors['bg_main'])
        self.ax = self.fig.add_subplot(111, facecolor=self.colors['bg_secondary'])
        self.ax.tick_params(colors=self.colors['text_primary'])
        self.ax.xaxis.label.set_color(self.colors['text_primary'])
        self.ax.yaxis.label.set_color(self.colors['text_primary'])
        self.ax.title.set_color(self.colors['text_primary'])
        self.ax.spines['bottom'].set_color(self.colors['text_secondary'])
        self.ax.spines['top'].set_color(self.colors['text_secondary'])
        self.ax.spines['right'].set_color(self.colors['text_secondary'])
        self.ax.spines['left'].set_color(self.colors['text_secondary'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Texto para mostrar estadísticas en GUI
        self.text_stats = tk.Text(frame_right, height=6, 
                                 bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                                 insertbackground=self.colors['text_primary'],
                                 selectbackground=self.colors['button_accent'],
                                 selectforeground=self.colors['text_primary'],
                                 font=('Consolas', 9), relief='flat', bd=0)
        self.text_stats.pack(fill=tk.X, pady=4)

    # -------------------------
    # Métodos de selección y UI
    # -------------------------
    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")])
        if not ruta:
            return
        # verificar existencia
        if not os.path.exists(ruta):
            messagebox.showerror("Error", f"No existe: {ruta}")
            return

        self.ruta_imagen = ruta
        # cargar imagen PIL para las funciones que la requieren (no modificamos)
        try:
            self.imagen_pil = Image.open(ruta)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen: {e}")
            return

        # mostrar preview en GUI
        self._mostrar_preview(self.imagen_pil)
        # al cargar, mostrar histogramas "Todos" por defecto en el canvas embebido
        self.mostrar_hist("Todos")

    def _mostrar_preview(self, imagen_pil):
        # ajusta tamaño para preview sin cambiar la imagen original en disco
        max_size = (360, 360)
        preview = imagen_pil.copy()
        preview.thumbnail(max_size)
        img_tk = ImageTk.PhotoImage(preview)
        self.panel_imagen.configure(image=img_tk)
        self.panel_imagen.image = img_tk

    # -------------------------
    # Botones que llaman a tus funciones originales
    # -------------------------
    def ejecutar_separar_rgb(self):
        if not self.imagen_pil:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        # llamada directa a la función original (mostrará sus plots con plt.show())
        separar_rgb(self.imagen_pil)

    def ejecutar_grises(self):
        if not self.imagen_pil:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        # uso de la función original (muestra su propia figura)
        gris = convertir_a_grises(self.imagen_pil)
        self.imagen_gris = gris
        # y también mostramos preview embebido (sin modificar la lógica de la función)
        try:
            img_pil = Image.fromarray(gris)
            self._mostrar_preview(img_pil)
        except Exception:
            # fallback: no romper si algo raro ocurre
            pass

    def ejecutar_binarizar(self):
        if self.imagen_gris is None:
            # intentar convertir a grises primero (llama la función original)
            if not self.imagen_pil:
                messagebox.showwarning("Aviso", "Primero carga una imagen.")
                return
            self.imagen_gris = convertir_a_grises(self.imagen_pil)
        # ahora llamar a la función original de binarización (muestra su propia figura)
        binarizar_imagen(self.imagen_gris)

    def ejecutar_cmyk(self):
        if not self.imagen_pil:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        convertir_a_cmyk(self.imagen_pil)

    def ejecutar_hsl(self):
        if not self.imagen_pil:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        convertir_a_hsl(self.imagen_pil)

    def ejecutar_binarizar(self):
        if self.imagen_gris is None:
            if not self.imagen_pil:
                messagebox.showwarning("Aviso", "Primero carga una imagen.")
                return
            self.imagen_gris = convertir_a_grises(self.imagen_pil)
        
        # Preguntar umbral al usuario
        umbral = simpledialog.askinteger("Umbral", "Ingrese un valor (0-255):", minvalue=0, maxvalue=255)
        if umbral is None:
            return  # usuario canceló
        binarizar_imagen_umbral(self.imagen_gris, umbral)

    def ejecutar_binarizar_otsu(self):
        if self.imagen_gris is None:
            if not self.imagen_pil:
                messagebox.showwarning("Aviso", "Primero carga una imagen.")
                return
            self.imagen_gris = convertir_a_grises(self.imagen_pil)
        binarizar_imagen_otsu(self.imagen_gris)

    # -------------------------
    # Histograma embebido (canvas)
    # -------------------------
    def mostrar_hist(self, canal):
        if not self.ruta_imagen or not os.path.exists(self.ruta_imagen):
            messagebox.showwarning("Aviso", "Primero carga una imagen válida.")
            return

        colores = {"Rojo": "red", "Verde": "green", "Azul": "blue"}
        self.ax.clear()
        
        # Reconfigurar el estilo del gráfico cada vez
        self.ax.set_facecolor(self.colors['bg_secondary'])
        self.ax.tick_params(colors=self.colors['text_primary'])
        self.ax.xaxis.label.set_color(self.colors['text_primary'])
        self.ax.yaxis.label.set_color(self.colors['text_primary'])
        self.ax.title.set_color(self.colors['text_primary'])
        self.ax.spines['bottom'].set_color(self.colors['text_secondary'])
        self.ax.spines['top'].set_color(self.colors['text_secondary'])
        self.ax.spines['right'].set_color(self.colors['text_secondary'])
        self.ax.spines['left'].set_color(self.colors['text_secondary'])
        self.ax.grid(True, color=self.colors['text_secondary'], alpha=0.3)

        if canal in ("Rojo", "Verde", "Azul"):
            hists = compute_histogramas_rgb_arrays(self.ruta_imagen)
            if hists is None:
                messagebox.showerror("Error", "No se pudo leer la imagen para histogramas.")
                return
            self.ax.plot(hists[canal], color=colores[canal], label=canal)
            self.ax.set_title(f"Histograma - {canal}")
        elif canal == "Todos":
            hists = compute_histogramas_rgb_arrays(self.ruta_imagen)
            if hists is None:
                messagebox.showerror("Error", "No se pudo leer la imagen para histogramas.")
                return
            self.ax.plot(hists['Rojo'], color='red', label='Rojo')
            self.ax.plot(hists['Verde'], color='green', label='Verde')
            self.ax.plot(hists['Azul'], color='blue', label='Azul')
            self.ax.set_title("Histograma - Todos")
            self.ax.legend()
        elif canal == "Grises":
            hist_gray = compute_histogram_gris_array(self.ruta_imagen)
            if hist_gray is None:
                messagebox.showerror("Error", "No se pudo leer la imagen en escala de grises.")
                return
            self.ax.plot(hist_gray, color='gray')
            self.ax.set_title("Histograma - Grises")
        else:
            self.ax.text(0.5, 0.5, "Canal desconocido", ha='center')

        self.ax.set_xlabel("Intensidad")
        self.ax.set_ylabel("Frecuencia")
        self.canvas.draw()

    # -------------------------
    # Mostrar estadísticas (opcional)
    # -------------------------
    def calcular_stats_via_funciones(self):
        """Llamar a las funciones originales que imprimen en consola."""
        if not self.ruta_imagen:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        # estas funciones abrirán sus propias figuras / imprimirán en consola
        calcular_histogramas_rgb(self.ruta_imagen)
        calcular_histograma_grises(self.ruta_imagen)

    def mostrar_stats_embebidos(self):
        """Calcular y mostrar estadísticas básicas embebidas (usando la misma lógica de cálculo)."""
        if not self.ruta_imagen:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        # calcular stats por canal (misma lógica de cálculos)
        imagen = cv2.imread(self.ruta_imagen)
        if imagen is None:
            messagebox.showerror("Error", "No se pudo leer la imagen.")
            return
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        resultados = {}
        for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
            datos = imagen_rgb[:, :, i].flatten()
            histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
            prob = histograma / histograma.sum()
            energia = np.sum(prob ** 2)
            entropia = -np.sum([p * log2(p) for p in prob if p > 0])
            asimetria = skew(datos)
            media = np.mean(datos)
            varianza = np.var(datos)
            resultados[canal] = {
                'Energía': energia,
                'Entropía': entropia,
                'Asimetría': asimetria,
                'Media': media,
                'Varianza': varianza
            }

        # mostrar en el text widget
        self.text_stats.delete("1.0", tk.END)
        for canal, props in resultados.items():
            self.text_stats.insert(tk.END, f"Canal {canal}:\n")
            for prop, valor in props.items():
                if isinstance(valor, float):
                    self.text_stats.insert(tk.END, f"  {prop}: {valor:.4f}\n")
                else:
                    self.text_stats.insert(tk.END, f"  {prop}: {valor}\n")
            self.text_stats.insert(tk.END, "\n")


# --------------------------
# EJECUTAR APP
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("1000x600")
    root.mainloop()