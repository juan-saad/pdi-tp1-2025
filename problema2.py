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
img_neg = cv2.bitwise_not(img_thresh)

fig, axes = plt.subplots(2, 2, constrained_layout=True, sharex=True, sharey=True)
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Imagen original")
axes[0, 0].axis("off")

axes[1, 0].imshow(img_thresh, cmap="gray")
axes[1, 0].set_title("Binaria + OTSU")
axes[1, 0].axis("off")

axes[0, 1].imshow(img_neg, cmap="gray")
axes[0, 1].set_title("Negativa")
axes[0, 1].axis("off")

axes[1, 1].axis("off")
plt.show(block=False)

# --- Calcular el Perfil de Proyección Horizontal ---
# Para detectar líneas de texto conviene contar los píxeles de texto por fila.
# Dependiendo de la binarización, el texto puede ser 0 (negro) sobre 255 (blanco)
# o al revés. Invertimos la binaria para que el texto valga 255 y el fondo 0,
# luego contamos píxeles no nulos por fila.

horizontal_projection = np.count_nonzero(img_neg == 255, axis=1)
index = np.argwhere(horizontal_projection > 900)

lista = []
ant = 9

for i in index.flatten():
    if i == 9 or i == 486:
        continue

    if i == ant + 1:
        start = i + 1
    else:
        end = i - 1
        lista.append((start, end))
    ant = i

vertical_projection = np.count_nonzero(img_neg == 255, axis=0)
vertical_projection

index_vert = np.argwhere(vertical_projection > 400)
index_si_no = np.argwhere(vertical_projection > 170)

index_vert
index_si_no

inicio_reg = 315
final_reg_medio = 616
inicio_reg_medio = 618
final_reg = 920

