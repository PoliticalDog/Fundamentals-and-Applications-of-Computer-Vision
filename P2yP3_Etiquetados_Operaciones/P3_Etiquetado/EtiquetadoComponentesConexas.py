import numpy as np  
from scipy import ndimage  
import matplotlib.pyplot as plt  

# Matriz, simula una imagen binaria
# 0 --> Fondo
# 1 --> Seccion de figura
imagen_binaria = np.array([  
 [0, 0, 0, 1, 1, 0, 0, 0],  
 [0, 1, 1, 1, 1, 1, 0, 0],  
 [0, 1, 1, 0, 0, 1, 1, 0],  
 [0, 0, 0, 1, 1, 0, 0, 0],  
 [0, 0, 1, 1, 0, 0, 1, 1],  
 [0, 1, 1, 1, 1, 1, 1, 0],  
 [0, 0, 0, 1, 0, 0, 0, 0]  
], dtype=int)  

# Definicion de vecindad 4 -
vecindad_4 = np.array([[0, 1, 0],  
                       [1, 1, 1],  
                       [0, 1, 0]], dtype=int)  

#Definicion de vecindad 8
vecindad_8 = np.ones((3, 3), dtype=int) # Matriz de 8-conexión  

# Etiquetados de vecindad
#recorre la imagen del array, y los procesa, regresa el array con mapeado de figuras
etiquetas_4, num_objetos_4 = ndimage.label(imagen_binaria, structure=vecindad_4) 
etiquetas_8, num_objetos_8 = ndimage.label(imagen_binaria, structure=vecindad_8) 

# Mostrar el número de objetos detectados  
print("Número de objetos con vecindad 4:", num_objetos_4)  
print("Número de objetos con vecindad 8:", num_objetos_8)  
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  

# Imagen binaria original  
axes[0].imshow(imagen_binaria, cmap='gray')  
axes[0].set_title("Imagen Binaria")  
axes[0].axis('off')  

# Etiquetado con vecindad 4  
axes[1].imshow(etiquetas_4, cmap='nipy_spectral')  
axes[1].set_title(f"Vecindad 4 - {num_objetos_4} Objetos")  
axes[1].axis('off') 
# Etiquetado con vecindad 8  
axes[2].imshow(etiquetas_8, cmap='nipy_spectral')  
axes[2].set_title(f"Vecindad 8 - {num_objetos_8} Objetos")  
axes[2].axis('off')  
plt.show()  
