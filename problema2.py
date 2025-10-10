from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

fig, axes = plt.subplots(2, 2, constrained_layout=True, sharex=True, sharey=True)
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Imagen original")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_thresh, cmap="gray")
axes[0, 1].set_title("Binaria + OTSU")
axes[0, 1].axis("off")

"""
La versión anterior de la negativa se basa en la comparación directa de píxeles.
La dejo para comparar con la version usando OpenCV.
"""
img_zeros = img != 255

axes[1, 0].imshow(img_zeros, cmap="gray")
axes[1, 0].set_title("Negativa con numpy")
axes[1, 0].axis("off")
axes[1, 1].axis("off")

plt.show(block=False)

# --- Calcular el Perfil de Proyección Horizontal ---
# Para detectar líneas de texto conviene contar los píxeles de texto por fila.
# Dependiendo de la binarización, el texto puede ser 0 (negro) sobre 255 (blanco)
# o al revés. Invertimos la binaria para que el texto valga 255 y el fondo 0,
# luego contamos píxeles no nulos por fila.

# Asegurarnos de que el texto valga 255
img_neg = cv2.bitwise_not(img_thresh)
plt.figure()
plt.imshow(img_neg, cmap="gray")
plt.title("Negativa")
plt.show(block=False)

horizontal_projection = np.count_nonzero(img_neg == 255, axis=1)

# El borde mas a la izquierda siempre se va a acercar a img_neg.shape[1] -> ~932
img_neg.shape[1]

# Suavizar la proyección para juntar pequeños ruidos y facilitar la detección
window = 15  # tamaño de ventana para el suavizado
kernel = np.ones(window, dtype=float) / window # Calculo del kernel usando la media movil

horizontal_smooth = np.convolve(horizontal_projection, kernel, mode="same")

fig, axes = plt.subplots(1, 2, constrained_layout=True)

axes[0].imshow(img_thresh, cmap="gray")
axes[0].set_title("Binaria + OTSU")

axes[1].plot(horizontal_projection, label="raw")
axes[1].plot(horizontal_smooth, label=f"smooth (w={window})")
axes[1].set_title("Perfil de Proyección Horizontal")
axes[1].legend()

plt.show(block=False)

# --- Encontrar las regiones que corresponden a líneas ---
# En lugar de usar la media, usar un percentil o una fracción del máximo suele
# ser más robusto cuando hay muchas filas en blanco.
# Opciones: np.percentile(horizontal_smooth, p) o factor * max
threshold = np.percentile(horizontal_smooth, 70)

# Binarizar el perfil según el umbral
above = horizontal_smooth > threshold

# Encontrar intervalos contiguos donde above == True
diff = np.diff(above.astype(int))

starts = np.where(diff == 1)[0] + 1
ends = np.where(diff == -1)[0] + 1

line_regions = list(zip(starts, ends))

# Visualizar los límites detectados en la imagen binaria
fig, ax = plt.subplots(constrained_layout=True)
ax.imshow(img_neg, cmap="gray")
ax.set_title("Regiones detectadas")
for start, end in line_regions:
    ax.axhline(y=start, color="red", linewidth=1)
    ax.axhline(y=end, color="red", linewidth=1)
plt.show(block=False)

# Calcular centros de línea para visualización
# line_centers = [int((s + e) / 2) for s, e in line_regions]

# # Dibujar las líneas detectadas sobre la imagen y marcar en el perfil
# fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 6))
# axes[0].imshow(img_thresh, cmap="gray")
# axes[0].set_title("Imagen original con líneas detectadas")
# for y in line_centers:
#     axes[0].axhline(y=y, color="red", linewidth=1)
# axes[0].axis("off")

# axes[1].plot(horizontal_smooth, label="smooth")
# axes[1].axhline(threshold, color="orange", linestyle="--", label="threshold")
# for y in line_centers:
#     axes[1].axvline(x=y, color="red", linewidth=1)
# axes[1].set_title("Proyección - líneas detectadas")
# axes[1].legend()

# plt.show(block=False)