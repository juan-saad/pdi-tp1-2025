from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE_DIR = Path(__name__).parent
IMAGE_PATH = BASE_DIR / "imagenes" / "formulario_01.png"

img = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {IMAGE_PATH}")

img.dtype

# Esto está para ayudar a pylance a inferir el tipo
img_thresh: np.ndarray

# Combina la bandera binaria con el algoritmo Otsu para hallar el umbral automáticamente.
ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(1, 2, constrained_layout=True, sharex=True, sharey=True)
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Imagen original")

axes[1].imshow(img_thresh, cmap="gray")
axes[1].set_title("Binaria + OTSU")

plt.show(block=False)

# Analisis de la imagen binarizada
# Se obtienen las proyecciones horizontal y vertical de la imagen binarizada
# Se suman los pixeles negros (0) en cada fila y columna
proyeccion_horizontal = np.sum(img_thresh == 0, axis=1)
proyeccion_vertical = np.sum(img_thresh == 0, axis=0)

# Se calcula el umbral para la proyección horizontal y vertical
umbral_horizontal = np.percentile(proyeccion_horizontal, 98)
umbral_vertical = np.percentile(proyeccion_vertical, 99.4)

# Obtenemos los índices donde la proyección supera o es igual al umbral
# El [0] es porque np.where devuelve una tupla
indices_lineas = np.where(proyeccion_horizontal >= umbral_horizontal)[0]
indices_columnas = np.where(proyeccion_vertical >= umbral_vertical)[0]

# Grafico para visualizar los valores de los umbrales definidos
fig, (ax_h, ax_v) = plt.subplots(1, 2, constrained_layout=True)

ax_h.plot(proyeccion_horizontal, label="horizontal_projection")
ax_h.axhline(umbral_horizontal, color="red", linestyle="--", label="th_horizontal")
ax_h.set_title("Proyección horizontal")
ax_h.legend()

ax_v.plot(proyeccion_vertical, label="vertical_projection")
ax_v.axhline(umbral_vertical, color="green", linestyle="--", label="th_vertical")
ax_v.set_title("Proyección vertical")
ax_v.legend()

plt.show(block=False)

# Deteccion de cambio de línea y columnas
# Se calcula la diferencia entre índices consecutivos
espacio_entre_lineas = np.diff(indices_lineas)
espacio_entre_columnas = np.diff(indices_columnas)

# Se consideran como saltos aquellos espacios mayores a 1, indican espacios en blanco entre líneas o columnas
salto_lineas = np.where(espacio_entre_lineas > 1)[0]
salto_columnas = np.where(espacio_entre_columnas > 1)[0]

# Se agrupan los índices en base a los saltos detectados
# Se suma 1 a los índices de salto para incluir el primer índice del nuevo grupo ya que el diff lo excluye
grupo_lineas = np.split(indices_lineas, salto_lineas + 1)
grupo_columnas = np.split(indices_columnas, salto_columnas + 1)

# Se calcula la media de cada grupo para obtener una única coordenada representativa
# Se convierte a entero para usar como coordenada de píxel
coordenadas_finales_lineas = [np.mean(grupo) for grupo in grupo_lineas]
coordenadas_finales_columnas = [np.mean(grupo) for grupo in grupo_columnas]

# Visualización de los resultados
fig, ax = plt.subplots(constrained_layout=True)

# Mostrar la imagen binarizada
ax.imshow(img_thresh, cmap="gray")

# Superponer las líneas horizontales detectadas
for y in coordenadas_finales_lineas:
    ax.axhline(float(y), color="red", linestyle="--", linewidth=2)

# Superponer las líneas verticales detectadas
for x in coordenadas_finales_columnas:
    ax.axvline(float(x), color="cyan", linestyle="--", linewidth=2)

ax.set_title("Detección de Cuadrícula Superpuesta")
ax.axis("off")
plt.show(block=False)

# Diccionario de tuplas con las coordenadas de las regiones del formulario
# Cada tupla tiene la forma (punto superior izquierdo, punto inferior derecho, punto superior izquierdo, punto inferior derecho)
formulario = {
    "nombre_apellido": (
        int(coordenadas_finales_lineas[1] + 1.5),
        int(coordenadas_finales_lineas[2] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
    "edad": (
        int(coordenadas_finales_lineas[2] + 1.5),
        int(coordenadas_finales_lineas[3] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
    "mail": (
        int(coordenadas_finales_lineas[3] + 1.5),
        int(coordenadas_finales_lineas[4] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
    "legajo": (
        int(coordenadas_finales_lineas[4] + 1.5),
        int(coordenadas_finales_lineas[5] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
    "pregunta_1_si": (
        int(coordenadas_finales_lineas[6] + 1.5),
        int(coordenadas_finales_lineas[7] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[2] - 1.5),
    ),
    "pregunta_1_no": (
        int(coordenadas_finales_lineas[6] + 1.5),
        int(coordenadas_finales_lineas[7] - 1.5),
        int(coordenadas_finales_columnas[2] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
    "pregunta_2_si": (
        int(coordenadas_finales_lineas[7] + 1.5),
        int(coordenadas_finales_lineas[8] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[2] - 1.5),
    ),
    "pregunta_2_no": (
        int(coordenadas_finales_lineas[7] + 1.5),
        int(coordenadas_finales_lineas[8] - 1.5),
        int(coordenadas_finales_columnas[2] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
    "pregunta_3_si": (
        int(coordenadas_finales_lineas[8] + 1.5),
        int(coordenadas_finales_lineas[9] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[2] - 1.5),
    ),
    "pregunta_3_no": (
        int(coordenadas_finales_lineas[8] + 1.5),
        int(coordenadas_finales_lineas[9] - 1.5),
        int(coordenadas_finales_columnas[2] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
    "comentarios": (
        int(coordenadas_finales_lineas[9] + 1.5),
        int(coordenadas_finales_lineas[10] - 1.5),
        int(coordenadas_finales_columnas[1] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
}

# Visualización de los centros de cada región del formulario
fig, ax = plt.subplots(constrained_layout=True)
ax.imshow(img_thresh, cmap="gray")

for campo, (y_min, y_max, x_min, x_max) in formulario.items():
    corners = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    ]
    for x_val, y_val in corners:
        ax.scatter(x_val, y_val, color="red", s=30)
    ax.text(x_min + 5, y_min + 15, campo, color="red", fontsize=8)

ax.set_title("Centros detectados del formulario")
ax.axis("off")
plt.show(block=False)

zona_interes = {}

for key in formulario.keys():
    y_min, y_max, x_min, x_max = formulario[key]
    zona_interes[key] = img_thresh[y_min:y_max, x_min:x_max]

# Visualización de las regiones de interés
n_celdas = len(zona_interes)
n_cols = min(3, n_celdas)
n_rows = int(np.ceil(n_celdas / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
if isinstance(axes, np.ndarray):
    ejes = axes.ravel()
else:
    ejes = [axes]

for ax, (campo, roi) in zip(ejes, zona_interes.items()):
    ax.imshow(roi, cmap="gray", vmin=0, vmax=255)
    ax.set_title(campo)
    ax.axis("off")

for ax in ejes[len(zona_interes) :]:
    ax.remove()

plt.show(block=False)


# TODO - Queda pendiente generar las funciones para validar caracteres y cantidad de palabras, se puede usar lo que tenemos abajo como ejemplo.
'''
for campo, roi in zona_interes.items():
    roi_binaria = (roi == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_binaria, connectivity=8, ltype=cv2.CV_32S)

    # celda_img es tu imagen binaria de entrada

    # 1. Definir un kernel rectangular para fusionar horizontalmente
    # El ancho (ej. 5 o 7) debe ser mayor que el espacio entre las letras.
    # El alto (ej. 1) asegura que no se fusionen líneas de texto cercanas.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))

    # 2. Aplicar la dilatación para conectar letras cercanas
    imagen_fusionada = cv2.dilate(roi_binaria, kernel, iterations=2)

    # 3. Aplicar connectedComponentsWithStats a la imagen fusionada
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_binaria, connectivity=8, ltype=cv2.CV_32S)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(imagen_fusionada, cmap="gray", vmin=0, vmax=255)
'''

# ... Su código de ConnectedComponentsWithStats

for campo, roi in zona_interes.items():
    roi_binaria = (roi == 0).astype(np.uint8)
    '''
    # 1. Dilatación para fusionar caracteres de una misma palabra (como ya tiene en su código)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    imagen_fusionada = cv2.dilate(roi_binaria, kernel, iterations=2)

    '''
    
    # **Usamos la imagen fusionada para encontrar palabras, no caracteres.**
    # Pero si quiere contar caracteres y luego palabras, use 'roi_binaria'
    # para 'connectedComponentsWithStats' y siga los pasos de aquí:
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_binaria, connectivity=8, ltype=cv2.CV_32S)

    # Solo analizar si se detectó más de un objeto (más allá del fondo)
    if num_labels > 1:
        # 1. Preparar y Ordenar los Centroides
        centroides_objetos = centroids[1:]
        centroides_ordenados = centroides_objetos[centroides_objetos[:, 0].argsort()]
        
        # 2. Calcular las Distancias Horizontales
        x_coordenadas = centroides_ordenados[:, 0]
        distancias_x = np.diff(x_coordenadas)
        
        # 3. Determinar un Umbral de Distancia
        mediana_distancia = np.median(distancias_x)
        umbral_distancia = mediana_distancia * 2.0  # Factor de 2.0 como punto de partida
        
        # 4. Identificar los Saltos de Palabra
        indices_saltos_palabra = np.where(distancias_x > umbral_distancia)[0]
        
        # El número de palabras es igual a la cantidad de saltos + 1
        numero_palabras = len(indices_saltos_palabra) + 1
        
        # El número de caracteres es la cantidad total de componentes (sin el fondo)
        numero_caracteres = num_labels - 1
        
        print(f"Campo: {campo}")
        print(f"  Caracteres detectados: {numero_caracteres}")
        print(f"  Palabras estimadas: {numero_palabras}")
        print(f"  Distancias entre centroides (fragmento): {distancias_x[:5]}")
        print(f"  Umbral de distancia (T): {umbral_distancia:.2f}")

    else:
        print(f"Campo: {campo}. No se detectaron objetos/caracteres.")

# ... Su código de visualización