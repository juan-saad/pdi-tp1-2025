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
max_index = np.max(horizontal_projection)
max_index
index = np.argwhere(horizontal_projection > 90*max_index/100)

lista = []
ant = index[0]
ult = index[-1]

for i in index.flatten():
    if i == ant or i == ult:
        continue

    if i == ant + 1:
        start = i + 1
    else:
        end = i - 1
        lista.append((start, end))
    ant = i


vertical_projection = np.count_nonzero(img_neg == 255, axis=0)

max_index_vert = np.max(vertical_projection)
index_si_no = np.argwhere(vertical_projection > 90*max_index_vert/100)
index_si_no

inicio_vert = index_si_no[2] + 2
final_vert = index_si_no[-1] - 1

lista.pop(5)
lista.pop(0)

def desempaque_coor(lista, inicio, final):
    dic = {}
    arriba, abajo = lista.pop(0)
    dic['inicio_izq'] = (arriba, inicio)
    dic['inicio_der'] = (arriba, final)
    dic['final_izq'] = (abajo, inicio)
    dic['final_der'] = (abajo, final)
    return dic

lista_campos = ["coor_ape_nomb", "coor_edad", "coor_mail", "coor_legajo", "coor_preg1", "coor_preg2", "coor_preg3", "coor_coment"]

diccionario_coordenadas = {}

for nombre_campo in lista_campos:
    diccionario_coordenadas[nombre_campo] = desempaque_coor(lista, inicio_vert, final_vert)
   

# Graficar puntos rojos en las coordenadas guardadas en el diccionario sobre img_thresh
plt.figure(figsize=(8, 8))
plt.imshow(img_thresh, cmap="gray")
plt.title("Coordenadas detectadas sobre la imagen binaria")
plt.axis("off")

for campo, coords in diccionario_coordenadas.items():
    for nombre, (y, x) in coords.items():
        plt.plot(x, y, 'ro')  # 'ro' = punto rojo
        plt.text(x + 5, y, campo, color='red', fontsize=8)  # Opcional: etiqueta

plt.show()

def recortar_por_coordenadas(img, coords):
    """
    Recorta una región de la imagen usando 4 coordenadas:
    'inicio_izq', 'inicio_der', 'final_izq', 'final_der'
    """
    y_inicio, x_inicio = coords['inicio_izq']
    y_fin, x_fin = coords['final_der']
    
   
    y1 = int(y_inicio)
    x1 = int(x_inicio)
    y2 = int(y_fin)
    x2 = int(x_fin)

    return img[y1:y2, x1:x2]


for campo, coor in diccionario_coordenadas.items():
    recorte = recortar_por_coordenadas(img, coor)
    plt.figure()
    plt.imshow(recorte, cmap="gray")
    plt.title(f"Recorte: {campo}")
    plt.axis("off")
    plt.show()


