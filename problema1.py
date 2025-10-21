from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def histograma(img: np.ndarray) -> None:
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


def mostrar_comparacion(
    img1: np.ndarray,
    img2: np.ndarray,
    titulo1: str,
    titulo2: str,
    figsize: Tuple[int, int] = (15, 6),
) -> None:
    """Muestra dos imágenes lado a lado para comparación."""
    _, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(img1, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(titulo1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(titulo2)
    axes[1].axis("off")

    plt.show(block=False)


def procesar_con_ventana(
    img: np.ndarray, M: int, N: int, mostrar_histograma: bool = True
) -> np.ndarray:
    """Aplica ecualización local con ventana MxN y opcionalmente muestra el histograma."""
    img_procesada = ecualizar_local(img, M, N)

    if mostrar_histograma:
        histograma(img_procesada)

    return img_procesada


def ecualizar_local(img_original: np.ndarray, M: int, N: int) -> np.ndarray:
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


def main() -> None:
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
    mostrar_comparacion(
        img_original,
        img_eq_global,
        "1. Imagen Original",
        "2. Imagen original ecualizada (global)",
    )

    histograma(img_eq_global)

    # Ecualizacion local con ventana 25x25 directamente sobre la imagen original
    img_procesada_25x25 = procesar_con_ventana(
        img_original, 25, 25, mostrar_histograma=False
    )
    mostrar_comparacion(
        img_original,
        img_procesada_25x25,
        "1. Imagen Original",
        "2. Imagen original ecualizada (local) con salt & pepper 25x25",
    )
    histograma(img_procesada_25x25)

    # Aplicar un filtro de la mediana para eliminar el ruido fino antes de la ecualización.
    # 3 es un tamaño de kernel pequeño que preserva los bordes.
    img_suavizada = cv2.medianBlur(img_original, 3)
    mostrar_comparacion(
        img_original, img_suavizada, "1. Imagen Original", "2. Imagen suavizada con k=3"
    )

    # Probar diferentes tamaños de ventana sobre la imagen suavizada
    tamaños_ventana = [5, 25, 75]

    for M in tamaños_ventana:
        N = M  # Ventana cuadrada
        img_procesada = procesar_con_ventana(
            img_suavizada, M, N, mostrar_histograma=True
        )
        mostrar_comparacion(
            img_original,
            img_procesada,
            "1. Imagen Original",
            f"2. Ecualización Local Manual (Ventana {M}x{N})",
        )


if __name__ == "__main__":
    main()
