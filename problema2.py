from calendar import c
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__name__).parent
IMAGE_PATH = BASE_DIR / "imagenes" / "formulario_vacio.png"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Combina la bandera binaria con Otsu para hallar el umbral automáticamente.
negativa = cv2.bitwise_not(thresh) # Bitwise_not invierte los valores de píxeles (0 a 255 y viceversa).

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Imagen original")
axes[0, 0].axis("off")

axes[0, 1].imshow(negativa, cmap="gray")
axes[0, 1].set_title("Negativa binaria")
axes[0, 1].axis("off")

"""
La versión anterior de la negativa se basa en la comparación directa de píxeles.
La dejo para comparar con la version usando OpenCV.
"""
img_zeros = img != 255

axes[1, 1].imshow(img_zeros, cmap="gray")
axes[1, 1].set_title("Negativa con numpy")
axes[1, 1].axis("off")

axes[1, 0].imshow(thresh, cmap="gray")
axes[1, 0].set_title(f"Umbral óptimo: {ret:.1f}")
axes[1, 0].axis("off")

plt.show(block=False)

img_row_zeros = img_zeros.any(axis=2)
img_row_zeros
plt.figure(), plt.plot(img_row_zeros), plt.show()

xr = img_row_zeros * (img.shape[1] - 1)
yr = np.arange(img.shape[0])
plt.figure(), plt.imshow(img, cmap="gray"), plt.plot(xr, yr, c="r"), plt.title(
    "Renglones"
), plt.show(block=False)
