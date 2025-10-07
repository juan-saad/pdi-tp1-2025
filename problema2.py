import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(
    "C:\\Users\\Administrador\\repos\\procesamiento de imagenes\\pdi-tp1-2025\\imagenes\\formulario_vacio.png",
    cv2.IMREAD_GRAYSCALE,
)
plt.figure(), plt.imshow(img, cmap="gray"), plt.show(block=False)

img_zeros = img != 255
plt.figure(), plt.imshow(img_zeros, cmap="gray"), plt.show()

img_row_zeros = img_zeros.any(axis=2)
img_row_zeros
plt.figure(), plt.plot(img_row_zeros), plt.show()

xr = img_row_zeros * (img.shape[1] - 1)
yr = np.arange(img.shape[0])
plt.figure(), plt.imshow(img, cmap="gray"), plt.plot(xr, yr, c="r"), plt.title(
    "Renglones"
), plt.show(block=False)
