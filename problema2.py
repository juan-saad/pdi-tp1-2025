from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__name__).parent
IMAGE_PATH = BASE_DIR / "imagenes" / "formulario_05.png"

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
    "tipo_formulario": (
        int(coordenadas_finales_lineas[0] + 1.5),
        int(coordenadas_finales_lineas[1] - 1.5),
        int(coordenadas_finales_columnas[0] + 1.5),
        int(coordenadas_finales_columnas[3] - 1.5),
    ),
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

# Prueba para ver los centroides en el campo "mail"
roi = zona_interes["mail"]
roi_binaria = (roi == 0).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_binaria, connectivity=8, ltype=cv2.CV_32S)

centroids[1:][:,0].argsort()

# Visualizar ROI con centroides
fig, ax = plt.subplots(constrained_layout=True)
ax.imshow(roi, cmap="gray", vmin=0, vmax=255)

for i in range(1, num_labels):
    cx, cy = centroids[i]
    ax.scatter(cx, cy, color="red", s=50, marker="x", linewidths=2)
    ax.text(cx + 3, cy + 3, str(i), color="red", fontsize=8)

ax.set_title(f"Centroides detectados: mail ({num_labels - 1} componentes)")
plt.show(block=False)

def calcular_caracteres(roi: np.ndarray) -> int:
    roi_binaria = (roi == 0).astype(np.uint8)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(roi_binaria, connectivity=8, ltype=cv2.CV_32S)
    return num_labels - 1

def calcular_palabras(roi: np.ndarray, campo: str) -> int:
    umbrales_espacios = {
        "nombre_apellido": 21,
        "edad": 21,
        "mail": 26,
        "legajo": 24
    }    
    
    roi_binaria = (roi == 0).astype(np.uint8)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(roi_binaria, connectivity=8, ltype=cv2.CV_32S)
    
    numero_palabras = 0

    if num_labels > 1:
        centroides_objetos = centroids[1:]
        centroides_ordenados = centroides_objetos[centroides_objetos[:, 0].argsort()]
        
        x_coordenadas = centroides_ordenados[:, 0]
        distancias_x = np.diff(x_coordenadas)
        
        indices_saltos_palabra = np.where(distancias_x > umbrales_espacios[campo])[0]
        
        numero_palabras = len(indices_saltos_palabra) + 1
    else:
        print(f"Campo: {campo}. No se detectaron objetos/caracteres.")

    return numero_palabras

resultado_validaciones = {
    "nombre_apellido": {
        "caracteres": calcular_caracteres(zona_interes["nombre_apellido"]) <= 25,
        "palabras": calcular_palabras(zona_interes["nombre_apellido"], "nombre_apellido") > 1,
    },
    "edad": {
        "caracteres": calcular_caracteres(zona_interes["edad"]) >= 2 and calcular_caracteres(zona_interes["edad"]) <= 3,
        "palabras": calcular_palabras(zona_interes["edad"], "edad") == 1,
    },
    "mail": {
        "caracteres": calcular_caracteres(zona_interes["mail"]) <= 25,
        "palabras": calcular_palabras(zona_interes["mail"], "mail") == 1,
    },
    "legajo": {
        "caracteres": calcular_caracteres(zona_interes["legajo"]) == 8,
        "palabras": calcular_palabras(zona_interes["legajo"], "legajo") == 1,
    },
    "pregunta_1": {
        "caracteres": calcular_caracteres(zona_interes["pregunta_1_si"]) + calcular_caracteres(zona_interes["pregunta_1_no"]) == 1,
        "palabras": True,
    },
    "pregunta_2": {
        "caracteres": calcular_caracteres(zona_interes["pregunta_2_si"]) + calcular_caracteres(zona_interes["pregunta_2_no"]) == 1,
        "palabras": True,
    },
    "pregunta_3": {
        "caracteres": calcular_caracteres(zona_interes["pregunta_3_si"]) + calcular_caracteres(zona_interes["pregunta_3_no"]) == 1,
        "palabras": True,
    },
    "comentarios": {
        "caracteres": calcular_caracteres(zona_interes["comentarios"]) <= 25,
        "palabras": calcular_caracteres(zona_interes["comentarios"]) >= 1, # Al menos un carácter determina una unica palabra
    },
}

for campo, resultados in resultado_validaciones.items():
    # Guardar un booleano para mantener el tipo homogéneo en el diccionario
    resultados["es_valido"] = all(resultados.values())
    print(f"Campo: {campo}, Resultado: {resultados["es_valido"]}")