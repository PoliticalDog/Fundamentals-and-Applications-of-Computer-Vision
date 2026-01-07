"""
Ejemplo práctico de uso del módulo P8_Segmentacion
Demuestra cómo usar las funciones del módulo de forma independiente
"""

import sys
sys.path.insert(0, r'c:\Users\godie\Desktop\IMAGE ANALYSIS\Tecnicas transformacion')

import cv2
import numpy as np
from modulos import P8_Segmentacion as p8
import os

def crear_imagen_ejemplo():
    """Crea una imagen de ejemplo con varios rectángulos"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 220  # Fondo gris claro
    
    # Dibujar varios rectángulos de diferentes tamaños y posiciones
    cv2.rectangle(img, (50, 50), (150, 150), (100, 150, 255), -1)    # Rojo-Magenta
    cv2.rectangle(img, (300, 100), (400, 200), (100, 150, 255), -1)  # Mismo color, diferente tamaño
    cv2.rectangle(img, (420, 250), (500, 330), (100, 150, 255), -1)  # Más pequeño
    
    # Agregar algunos círculos para ruido
    cv2.circle(img, (100, 300), 30, (50, 100, 200), -1)
    cv2.circle(img, (500, 350), 40, (50, 100, 200), -1)
    
    return img

def crear_plantilla(img):
    """Extrae una plantilla de la imagen"""
    # Tomar una región de la imagen como plantilla
    plantilla = img[50:150, 50:150].copy()
    return plantilla

def ejemplo_basico():
    """Ejemplo 1: Template Matching Simple"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Template Matching Simple")
    print("="*60)
    
    # Crear imágenes de ejemplo
    imagen = crear_imagen_ejemplo()
    plantilla = crear_plantilla(imagen)
    
    print(f"Imagen: {imagen.shape}")
    print(f"Plantilla: {plantilla.shape}")
    
    # Realizar búsqueda
    resultado, matches = p8.template_matching(
        imagen, plantilla,
        method='TM_CCOEFF_NORMED',
        threshold=0.8
    )
    
    print(f"\nCoincidencias encontradas: {len(matches)}")
    if matches:
        print("Primeras 3 coincidencias:")
        for i, (x, y, valor) in enumerate(matches[:3]):
            print(f"  {i+1}. Posición ({x}, {y}) - Similitud: {valor:.4f}")
    
    # Guardar resultado
    output_dir = r'c:\Users\godie\Desktop\IMAGE ANALYSIS\Imagenes Resultados'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'P8_ejemplo1_simple.png')
    cv2.imwrite(output_path, resultado)
    print(f"\nResultado guardado en: {output_path}")

def ejemplo_multiscale():
    """Ejemplo 2: Template Matching Multiscala"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Template Matching Multiscala")
    print("="*60)
    
    # Crear imágenes de ejemplo
    imagen = crear_imagen_ejemplo()
    plantilla = crear_imagen_ejemplo()[50:150, 50:150].copy()  # Plantilla pequeña
    
    # Realizar búsqueda multiscala
    escalas = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    resultado, matches = p8.template_matching_multiscale(
        imagen, plantilla,
        scales=escalas,
        method='TM_CCOEFF_NORMED',
        threshold=0.75
    )
    
    print(f"Escalas probadas: {escalas}")
    print(f"\nCoincidencias encontradas: {len(matches)}")
    
    # Agrupar por escala
    por_escala = {}
    for x, y, val, scale in matches:
        if scale not in por_escala:
            por_escala[scale] = 0
        por_escala[scale] += 1
    
    print("\nCoincidencias por escala:")
    for scale in escalas:
        count = por_escala.get(scale, 0)
        print(f"  Escala {scale}: {count} coincidencias")
    
    # Guardar resultado
    output_dir = r'c:\Users\godie\Desktop\IMAGE ANALYSIS\Imagenes Resultados'
    output_path = os.path.join(output_dir, 'P8_ejemplo2_multiscale.png')
    cv2.imwrite(output_path, resultado)
    print(f"\nResultado guardado en: {output_path}")

def ejemplo_features():
    """Ejemplo 3: Feature Matching"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Feature Matching (SIFT)")
    print("="*60)
    
    # Crear imágenes de ejemplo
    imagen = crear_imagen_ejemplo()
    plantilla = crear_imagen_ejemplo()[50:150, 50:150].copy()
    
    # Realizar feature matching
    resultado, num_features = p8.feature_matching(
        imagen, plantilla,
        method='SIFT'
    )
    
    print(f"Características coincidentes encontradas: {num_features}")
    
    # Guardar resultado
    output_dir = r'c:\Users\godie\Desktop\IMAGE ANALYSIS\Imagenes Resultados'
    output_path = os.path.join(output_dir, 'P8_ejemplo3_features.png')
    cv2.imwrite(output_path, resultado)
    print(f"Resultado guardado en: {output_path}")

def ejemplo_contours():
    """Ejemplo 4: Contour Matching"""
    print("\n" + "="*60)
    print("EJEMPLO 4: Contour Matching")
    print("="*60)
    
    # Crear imágenes de ejemplo
    imagen = crear_imagen_ejemplo()
    plantilla = crear_imagen_ejemplo()[50:150, 50:150].copy()
    
    # Realizar contour matching
    resultado, matches = p8.contour_matching(
        imagen, plantilla,
        threshold=0.6
    )
    
    print(f"Contornos similares encontrados: {len(matches)}")
    
    # Guardar resultado
    output_dir = r'c:\Users\godie\Desktop\IMAGE ANALYSIS\Imagenes Resultados'
    output_path = os.path.join(output_dir, 'P8_ejemplo4_contours.png')
    cv2.imwrite(output_path, resultado)
    print(f"Resultado guardado en: {output_path}")

def main():
    print("\n" + "#"*60)
    print("# EJEMPLOS DE USO - P8_Segmentacion")
    print("#"*60)
    
    try:
        ejemplo_basico()
        ejemplo_multiscale()
        ejemplo_features()
        ejemplo_contours()
        
        print("\n" + "#"*60)
        print("# TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE")
        print("#"*60)
        print("\nResultados guardados en: c:\\Users\\godie\\Desktop\\IMAGE ANALYSIS\\Imagenes Resultados")
        
    except Exception as e:
        print(f"\n✗ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
