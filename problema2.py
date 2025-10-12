from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__name__).parent
IMAGE_PATH = BASE_DIR / "imagenes" / "formulario_vacio.png"

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
coordenadas_finales_lineas = [int(np.mean(grupo)) for grupo in grupo_lineas]
coordenadas_finales_columnas = [int(np.mean(grupo)) for grupo in grupo_columnas]

# Visualización de los resultados
fig, ax = plt.subplots(constrained_layout=True)

# Mostrar la imagen binarizada
ax.imshow(img_thresh, cmap="gray")

# Superponer las líneas horizontales detectadas
for y in coordenadas_finales_lineas:
    ax.axhline(y, color="red", linestyle="--", linewidth=2)

# Superponer las líneas verticales detectadas
for x in coordenadas_finales_columnas:
    ax.axvline(x, color="cyan", linestyle="--", linewidth=2)

ax.set_title("Detección de Cuadrícula Superpuesta")
ax.axis("off")
plt.show(block=False)

formulario = {
    "nombre_apellido": (
        coordenadas_finales_lineas[1],
        coordenadas_finales_lineas[2],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[3],
    ),
    "edad": (
        coordenadas_finales_lineas[2],
        coordenadas_finales_lineas[3],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[3],
    ),
    "mail": (
        coordenadas_finales_lineas[3],
        coordenadas_finales_lineas[4],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[3],
    ),
    "legajo": (
        coordenadas_finales_lineas[4],
        coordenadas_finales_lineas[5],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[3],
    ),
    "pregunta_1_si": (
        coordenadas_finales_lineas[5],
        coordenadas_finales_lineas[6],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[2],
    ),
    "pregunta_1_no": (
        coordenadas_finales_lineas[5],
        coordenadas_finales_lineas[6],
        coordenadas_finales_columnas[2],
        coordenadas_finales_columnas[3],
    ),
    "pregunta_2_si": (
        coordenadas_finales_lineas[6],
        coordenadas_finales_lineas[7],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[2],
    ),
    "pregunta_2_no": (
        coordenadas_finales_lineas[6],
        coordenadas_finales_lineas[7],
        coordenadas_finales_columnas[2],
        coordenadas_finales_columnas[3],
    ),
    "pregunta_3_si": (
        coordenadas_finales_lineas[7],
        coordenadas_finales_lineas[8],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[2],
    ),
    "pregunta_3_no": (
        coordenadas_finales_lineas[7],
        coordenadas_finales_lineas[8],
        coordenadas_finales_columnas[2],
        coordenadas_finales_columnas[3],
    ),
    "comentarios": (
        coordenadas_finales_lineas[8],
        coordenadas_finales_lineas[10],
        coordenadas_finales_columnas[1],
        coordenadas_finales_columnas[3],
    ),
}

formulario
