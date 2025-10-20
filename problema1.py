from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def histograma(img):
    """Muestra la imagen original, su histograma de frecuencias y el histograma normalizado con su funcion de suma acumulada."""

    # Mostrar la imagen
    # Mostrar la imagen y su histograma lado a lado
    _, axes = plt.subplots(1, 3, constrained_layout=True)

    # Imagen
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[0].set_title("Imagen")

    # Histograma
    hist, bins = np.histogram(img.flatten(), 256, (0, 256))
    bin_centers = bins[:-1]

    bar_width = 10

    axes[1].bar(bin_centers, hist, width=bar_width, color="black")
    axes[1].set_xlim([-50, 255])
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Intensidad")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Histograma")

    histn = hist.astype(np.double) / img.size
    cdf = histn.cumsum()

    # Histograma normalizado
    axes[2].bar(bin_centers, histn, width=bar_width, color="black")
    axes[2].set_xlim([-50, 255])
    axes[2].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Intensidad")
    axes[2].set_ylabel("Probabilidad (densidad)")
    axes[2].set_title("Histograma Normalizado")

    # Suma acumulada
    ax2 = axes[2].twinx()
    ax2.plot(bin_centers, cdf, color="red", linewidth=1.5, label="CDF")
    ax2.set_ylabel("CDF (acumulado)")
    ax2.set_ylim([0, 1])

    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.show(block=False)


def ecualizar_local(img_original, M, N):
    """Aplica ecualización de histograma local con ventana MxN usando cv2.equalizeHist en cada vecindario."""

    if M % 2 == 0 or N % 2 == 0:
        raise ValueError(
            "El tamaño de la ventana (M, N) debe ser impar para tener un centro."
        )

    # Se obtienen las dimenciones de la imagen de entrada
    H, W = img_original.shape

    img_ecualizada = np.zeros_like(img_original, dtype=np.uint8)

    m_pad = M // 2
    n_pad = N // 2

    # Agregar bordes a la imagen
    img_con_borde = cv2.copyMakeBorder(
        img_original, m_pad, m_pad, n_pad, n_pad, cv2.BORDER_REPLICATE
    )

    # Bucle Principal: Ventana Deslizante
    for i in range(H):
        for j in range(W):
            # 2.1. Definir la Ventana (Vecindario)
            ventana = img_con_borde[i : i + M, j : j + N]

            # 2.2. Aplicar cv2.equalizeHist() a la ventana
            # Esto produce una imagen pequeña (MxN) donde todos los píxeles están ecualizados
            ventana_ecualizada = cv2.equalizeHist(ventana)

            # 2.3. Obtener el Píxel Central Transformado

            # En la ventana_ecualizada, el píxel central (m_pad, n_pad)
            # es el resultado de la transformación para el píxel (i, j)
            nuevo_valor = ventana_ecualizada[m_pad, n_pad]

            # Asignar el nuevo valor a la imagen de salida
            img_ecualizada[i, j] = nuevo_valor

    return img_ecualizada


def main():
    """Programa principal: carga imagen, aplica ecualización global y local con diferentes tamaños de ventana."""

    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

    IMAGE_PATH = BASE_DIR / "imagenes" / "Imagen_con_detalles_escondidos.tif"

    # Cargar la imagen en escala de grises
    img_original = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if img_original is None:
        raise Exception(f"Error: No se pudo cargar la imagen en la ruta: {IMAGE_PATH}")

    histograma(img_original)

    # Ecualización Global con OpenCV
    img_eq_global = cv2.equalizeHist(img_original)

    # Mostrar original y equalizada global lado a lado
    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_eq_global, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Imagen original ecualizada (global)")
    axes[1].axis("off")

    plt.show(block=False)

    # ---------------------------------------------------------------------------------------------------

    # Parámetros (M y N deben ser impares)
    M = 25  # Alto de la ventana
    N = 25  # Ancho de la ventana

    # Aplicar la función directamente a la imagen original con ruido
    # Ecualizacion local con ventana 25x25
    img_procesada = ecualizar_local(img_original, M, N)

    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_procesada, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"2. Imagen original ecualizada (local) con salt & pepper {M}x{N}"
    )
    axes[1].axis("off")

    plt.show(block=False)

    # Mostrar el histograma de la imagen procesada
    histograma(img_procesada)

    # Aplicar un filtro de la mediana para eliminar el ruido fino antes de la ecualización.
    # 3 es un tamaño de kernel pequeño que preserva los bordes.
    img_suavizada = cv2.medianBlur(img_original, 3)

    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_suavizada, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Imagen suavizada con k=3")
    axes[1].axis("off")

    plt.show(block=False)

    # ---------------------------------------------------------------------------------------------------

    # VENTANA DE 5X5 (TAMAÑO CHICO)

    # Parámetros (M y N deben ser impares)
    M = 5  # Alto de la ventana
    N = 5  # Ancho de la ventana

    # Aplicar la función
    img_procesada = ecualizar_local(img_suavizada, M, N)

    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_procesada, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Ecualización Local Manual (Ventana {M}x{N})")
    axes[1].axis("off")

    plt.show(block=False)

    # Mostrar el histograma de la imagen procesada
    histograma(img_procesada)

    # ---------------------------------------------------------------------------------------------------

    # VENTANA DE 25X25 (TAMAÑO MEDIANO)

    # Parámetros (M y N deben ser impares)
    M = 25  # Alto de la ventana
    N = 25  # Ancho de la ventana

    # Aplicar la función
    img_procesada = ecualizar_local(img_suavizada, M, N)

    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_procesada, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Ecualización Local Manual (Ventana {M}x{N})")
    axes[1].axis("off")

    plt.show(block=False)

    # Mostrar el histograma de la imagen procesada
    histograma(img_procesada)

    # -------------------------------------------------------------------------------------

    # VENTANA DE 75X75 (TAMAÑO GRANDE)

    # Parámetros (M y N deben ser impares)
    M = 75  # Alto de la ventana
    N = 75  # Ancho de la ventana

    # Aplicar la función
    img_procesada = ecualizar_local(img_suavizada, M, N)

    # Mostrar los resultados
    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_procesada, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Ecualización Local Manual (Ventana {M}x{N})")
    axes[1].axis("off")

    plt.show(block=False)

    # Mostrar el histograma de la imagen procesada
    histograma(img_procesada)


if __name__ == "__main__":
    main()
