from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
def ecualizar_local_manual(img_original, M, N):
    """
    Implementa la Ecualización Local del Histograma manualmente con ventana deslizante.

    Parámetros:
    - img_original (np.array): La imagen en escala de grises.
    - M (int): Alto de la ventana (debe ser impar).
    - N (int): Ancho de la ventana (debe ser impar).

    Retorna:
    - np.array: La imagen procesada.
    """
    if M % 2 == 0 or N % 2 == 0:
        raise ValueError("El tamaño de la ventana (M, N) debe ser impar para tener un centro.")

    # 1. Preparación de la Imagen y Bordes (Padding)
    
    H, W = img_original.shape
    img_ecualizada = np.zeros_like(img_original, dtype=np.uint8)

    # Calcular el radio de la ventana (mitad del tamaño de la ventana)
    m_pad = M // 2
    n_pad = N // 2

    # Agregar bordes a la imagen para que la ventana pueda centrarse en los píxeles del borde.
    # Usamos cv2.BORDER_REPLICATE para replicar los valores del borde, como se sugirió.
    img_con_borde = cv2.copyMakeBorder(
        img_original,
        m_pad, m_pad, n_pad, n_pad,
        cv2.BORDER_REPLICATE
    )
    
    # 2. Bucle Principal: Ventana Deslizante
    
    # Iterar sobre cada píxel de la imagen original (sin bordes)
    for i in range(H):
        for j in range(W):
            # 2.1. Definir la Ventana (Vecindario)
            
            # Las coordenadas deben estar en la imagen con bordes
            # La ventana va desde (i) hasta (i + M) y desde (j) hasta (j + N) en img_con_borde
            ventana = img_con_borde[i : i + M, j : j + N]
            
            # 2.2. Calcular el Histograma y la Función de Transformación (T)
            
            # El número total de píxeles en la ventana
            num_pixeles_ventana = M * N
            
            # Crear el histograma (frecuencia de cada nivel de intensidad)
            hist, _ = np.histogram(ventana.flatten(), 256, [0, 256])
            
            # Calcular el Histograma Acumulado Normalizado (Función de Transformación T)
            # T[k] = (Suma acumulada de hist hasta k) * (L-1) / (M*N)
            cdf = hist.cumsum() # Cumulative Distribution Function (Suma Acumulada)
            
            # Normalizar: Ignorar cdf[0] si es cero para evitar división por cero.
            cdf_min = cdf.min() if cdf.min() > 0 else 0
            
            # Fórmula de Ecualización de Histograma
            # La transformación mapea cada nivel de intensidad a un nuevo nivel
            # [L-1 = 255 para 8 bits]
            T = np.round((cdf - cdf_min) * 255 / (num_pixeles_ventana - cdf_min)).astype('uint8')
            
            # 2.3. Aplicar la Transformación al Píxel Central
            
            # El píxel central de la ventana deslizante es el píxel (i, j) de la imagen original.
            intensidad_original = img_original[i, j]
            
            # El nuevo valor es el valor en la posición 'intensidad_original' de la función T
            nuevo_valor = T[intensidad_original]
            
            # Asignar el nuevo valor a la imagen de salida
            img_ecualizada[i, j] = nuevo_valor

    return img_ecualizada

'''
def ecualizar_local (img_original, M, N):
    """
    Implementa la Ecualización Local del Histograma con ventana deslizante
    utilizando cv2.equalizeHist() en cada vecindario.

    """
    if M % 2 == 0 or N % 2 == 0:
        raise ValueError("El tamaño de la ventana (M, N) debe ser impar para tener un centro.")

    # Se obtienen las dimenciones de la imagen de entrada
    H, W = img_original.shape

    img_ecualizada = np.zeros_like(img_original, dtype=np.uint8)

    m_pad = M // 2
    n_pad = N // 2

    # Agregar bordes a la imagen 
    img_con_borde = cv2.copyMakeBorder(
        img_original,
        m_pad, m_pad, n_pad, n_pad,
        cv2.BORDER_REPLICATE
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

# -----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------

BASE_DIR = Path(__name__).parent
IMAGE_PATH = BASE_DIR  / "pdi-tp1-2025" / "imagenes" / "Imagen_con_detalles_escondidos.tif"

'''
# Muestro la imágen original
img_original_normal = cv2.imread(IMAGE_PATH)


plt.imshow(img_original_normal)
plt.axis('off')
plt.show()

'''

# Cargar la imagen en escala de grises
img_original = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

'''
img_original = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

'''

if img_original is None:
    print(f"Error: No se pudo cargar la imagen en la ruta: {IMAGE_PATH}")

else:
    # Muestro la imágen original en escala de grises
    plt.imshow(img_original)
    plt.axis('off')
    plt.show()

    # Aplicar un filtro de la mediana para eliminar el ruido fino antes de la ecualización.
    # 3 es un tamaño de kernel pequeño que preserva los bordes.
    img_suavizada = cv2.medianBlur(img_original, 3) 

    # Muestro la imágen suavizada
    plt.imshow(img_suavizada, cmap='gray'); plt.axis('off'); plt.title('Imagen Suavizada'); plt.show()

    #---------------------------------------------------------------------------------------------------

    # VENTANA DE 25X25 (TAMAÑO MEDIANO)

    # Parámetros (M y N deben ser impares)
    M = 25 # Alto de la ventana
    N = 25 # Ancho de la ventana
    '''
    print(f"Iniciando ecualización local manual con ventana {M}x{N}. Esto puede tomar tiempo...")
    '''
    # Aplicar la función
    img_procesada = ecualizar_local(img_suavizada, M, N)
    
    print("¡Procesamiento completado!")

    # Mostrar los resultados
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('1. Imagen Original')
    axes[0].axis('off')

    axes[1].imshow(img_procesada, cmap='gray')
    axes[1].set_title(f'2. Ecualización Local Manual (Ventana {M}x{N})')
    axes[1].axis('off')

    plt.suptitle('Problema 1 - Ecualización Local de Histograma Manual', fontsize=16)
    plt.show()

    #-------------------------------------------------------------------------------------------

    # VENTANA DE 5X5 (TAMAÑO CHICO)

    # Parámetros (M y N deben ser impares)
    M = 5 # Alto de la ventana
    N = 5 # Ancho de la ventana
    '''
    print(f"Iniciando ecualización local manual con ventana {M}x{N}. Esto puede tomar tiempo...")
    '''
    # Aplicar la función
    img_procesada = ecualizar_local(img_suavizada, M, N) 
    
    print("¡Procesamiento completado!")

    # Mostrar los resultados
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('1. Imagen Original')
    axes[0].axis('off')

    axes[1].imshow(img_procesada, cmap='gray')
    axes[1].set_title(f'2. Ecualización Local Manual (Ventana {M}x{N})')
    axes[1].axis('off')

    plt.suptitle('Problema 1 - Ecualización Local de Histograma Manual', fontsize=16)
    plt.show()

    #-------------------------------------------------------------------------------------

    # VENTANA DE 75X75 (TAMAÑO GRANDE)

    # Parámetros (M y N deben ser impares)
    M = 75 # Alto de la ventana
    N = 75 # Ancho de la ventana
    '''
    print(f"Iniciando ecualización local manual con ventana {M}x{N}. Esto puede tomar tiempo...")
    '''
    # Aplicar la función
    img_procesada = ecualizar_local(img_suavizada, M, N) 
    
    print("¡Procesamiento completado!")

    # Mostrar los resultados
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('1. Imagen Original')
    axes[0].axis('off')

    axes[1].imshow(img_procesada, cmap='gray')
    axes[1].set_title(f'2. Ecualización Local Manual (Ventana {M}x{N})')
    axes[1].axis('off')

    plt.suptitle('Problema 1 - Ecualización Local de Histograma Manual', fontsize=16)
    plt.show()