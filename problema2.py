from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict
import csv


def _centro_valido(centroides: np.ndarray) -> Tuple[int, int]:
    """Calcula un centro válido a partir de los centroides detectados. Usa el último centroide ordenado por X, o el centroide del fondo si la imagen no tiene componentes válidos."""

    centros = centroides[centroides[:, 0].argsort()]
    cx = float(centros[-1][0])
    cy = float(centros[-1][1])

    if np.isnan(cx) or np.isnan(cy) or (cx == 0 and cy == 0):
        cx = centros[0][0]
        cy = centros[0][1]

    return (int(cx), int(cy))


def generar_roi(
    coordenadas: dict[str, tuple[int, int, int, int]],
    img: np.ndarray,
    mostrar_graficos: bool = True,
) -> dict[str, np.ndarray]:
    """Extrae las regiones de interés (ROI) de la imagen según las coordenadas dadas."""

    zona_interes = {}

    for key in coordenadas.keys():
        y_min, y_max, x_min, x_max = coordenadas[key]
        zona_interes[key] = img[y_min:y_max, x_min:x_max]

    # Visualización de las regiones de interés
    n_celdas = len(zona_interes)
    n_cols = min(3, n_celdas)
    n_rows = int(np.ceil(n_celdas / n_cols))

    if mostrar_graficos:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True
        )
        fig.suptitle("Regiones de interés detectadas", fontsize=16)

        # axes es de tipo numpy.ndarray
        ejes = axes.flatten()

        for ax, (campo, roi) in zip(ejes, zona_interes.items()):
            ax.imshow(roi, cmap="gray", vmin=0, vmax=255)
            ax.set_title(campo)
            ax.axis("off")

        for ax in ejes[len(zona_interes) :]:
            ax.remove()

        plt.show(block=False)

    return zona_interes


def dibujar_check(img: np.ndarray, centro: Tuple[int, int], grosor=2) -> np.ndarray:
    """Dibuja una marca de verificación (check) en la imagen en la posición indicada."""

    x, y = centro
    offset = 15

    cv2.line(img, (x - offset, y), (x, y + offset), 0, grosor)
    cv2.line(img, (x, y + offset), (x + offset, y - offset), 0, grosor)
    return img


def dibujar_x(img: np.ndarray, centro: Tuple[int, int], grosor=2) -> np.ndarray:
    """Dibuja una X en la imagen en la posición indicada."""

    x, y = centro
    offset = 15

    cv2.line(img, (x - offset, y - offset), (x + offset, y + offset), 0, grosor)
    cv2.line(img, (x - offset, y + offset), (x + offset, y - offset), 0, grosor)
    return img


def calcular_caracteres(roi: np.ndarray) -> int:
    """Cuenta el número de componentes conectados (caracteres) en la región de interés."""

    roi_binaria = (roi == 0).astype(np.uint8)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        roi_binaria, connectivity=8, ltype=cv2.CV_32S
    )
    return num_labels - 1


def buscar_coordenadas_formulario(
    img: np.ndarray, mostrar_graficos: bool = True
) -> dict[str, tuple[int, int, int, int]]:
    """Detecta las coordenadas de los campos del formulario usando proyecciones horizontal y vertical."""

    # Analisis de la imagen binarizada
    # Se obtienen las proyecciones horizontal y vertical de la imagen binarizada
    # Se suman los pixeles negros (0) en cada fila y columna
    proyeccion_horizontal = np.sum(img == 0, axis=1)
    proyeccion_vertical = np.sum(img == 0, axis=0)

    # Se calcula el umbral para la proyección horizontal y vertical
    umbral_horizontal = np.percentile(proyeccion_horizontal, 98)
    umbral_vertical = np.percentile(proyeccion_vertical, 99.4)

    # Obtenemos los índices donde la proyección supera o es igual al umbral
    # El [0] es porque np.where devuelve una tupla
    indices_lineas = np.where(proyeccion_horizontal >= umbral_horizontal)[0]
    indices_columnas = np.where(proyeccion_vertical >= umbral_vertical)[0]

    if mostrar_graficos:
        # Grafico para visualizar los valores de los umbrales definidos
        _, (ax_h, ax_v) = plt.subplots(1, 2, constrained_layout=True)

        ax_h.plot(proyeccion_horizontal, label="horizontal_projection")
        ax_h.axhline(
            umbral_horizontal, color="red", linestyle="--", label="th_horizontal"
        )
        ax_h.set_title("Proyección horizontal")
        ax_h.legend()

        ax_v.plot(proyeccion_vertical, label="vertical_projection")
        ax_v.axhline(
            umbral_vertical, color="green", linestyle="--", label="th_vertical"
        )
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

    if mostrar_graficos:
        # Visualización de los resultados
        _, ax = plt.subplots(constrained_layout=True)

        # Mostrar la imagen binarizada
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)

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

    if mostrar_graficos:
        # Visualización de los centros de cada región del formulario
        _, ax = plt.subplots(constrained_layout=True)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)

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

        ax.set_title("Coordenadas detectadas del formulario")
        ax.axis("off")
        plt.show(block=False)

    return formulario


def generar_imagen_binarizada(
    img: np.ndarray, mostrar_graficos: bool = True
) -> np.ndarray:
    """Binariza la imagen usando el método de Otsu para calcular el umbral automáticamente."""

    # Esto está para ayudar a pylance a inferir el tipo
    img_thresh: np.ndarray

    # Combina la bandera binaria con el algoritmo Otsu para hallar el umbral automáticamente.
    val, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"Umbral calculado por OTSU: {val}\n")

    if mostrar_graficos:
        _, axes = plt.subplots(1, 2, constrained_layout=True, sharex=True, sharey=True)
        axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Imagen original")

        axes[1].imshow(img_thresh, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Binaria + OTSU")

        plt.show(block=False)

    return img_thresh


def calcular_palabras(roi: np.ndarray, campo: str) -> int:
    """Cuenta el número de palabras en la región de interés según distancias entre centroides."""

    umbrales_espacios = {"nombre_apellido": 21, "edad": 21, "mail": 26, "legajo": 24}

    roi_binaria = (roi == 0).astype(np.uint8)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(
        roi_binaria, connectivity=8, ltype=cv2.CV_32S
    )

    numero_palabras = 0

    if num_labels > 1:
        centroides_objetos = centroids[1:]
        centroides_ordenados = centroides_objetos[centroides_objetos[:, 0].argsort()]

        x_coordenadas = centroides_ordenados[:, 0]

        # Con las distancias en X calculamos los saltos entre palabras
        distancias_x = np.diff(x_coordenadas)

        indices_saltos_palabra = np.where(distancias_x > umbrales_espacios[campo])[0]

        numero_palabras = len(indices_saltos_palabra) + 1
    else:
        print(f"Campo: {campo}. No se detectaron objetos/caracteres.")

    return numero_palabras


def analisis_formulario(
    img: np.ndarray, mostrar_graficos: bool = True
) -> Tuple[Dict[str, Dict[str, bool]], bool]:
    """Analiza un formulario completo, validando cada campo según criterios específicos de caracteres y palabras."""

    img.dtype
    img.dtype

    img_thresh = generar_imagen_binarizada(img, mostrar_graficos)

    formulario = buscar_coordenadas_formulario(img_thresh, mostrar_graficos)

    zona_interes = generar_roi(formulario, img_thresh, mostrar_graficos)

    # Prueba para ver los centroides en el campo "mail"
    roi = zona_interes["mail"]
    roi_binaria = (roi == 0).astype(np.uint8)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(
        roi_binaria, connectivity=8, ltype=cv2.CV_32S
    )

    if mostrar_graficos:
        # Visualizar ROI con centroides
        _, ax = plt.subplots(constrained_layout=True)
        ax.imshow(roi, cmap="gray", vmin=0, vmax=255)

        for i in range(1, num_labels):
            cx, cy = centroids[i]
            ax.scatter(cx, cy, color="red", s=50, marker="x", linewidths=2)
            ax.text(cx + 3, cy + 3, str(i), color="red", fontsize=8)

        ax.set_title(f"Centroides detectados: mail ({num_labels - 1} componentes)")
        plt.show(block=False)

    resultado_validaciones = {
        "nombre_apellido": {
            "caracteres": calcular_caracteres(zona_interes["nombre_apellido"]) <= 25,
            "palabras": calcular_palabras(
                zona_interes["nombre_apellido"], "nombre_apellido"
            )
            > 1,
        },
        "edad": {
            "caracteres": calcular_caracteres(zona_interes["edad"]) >= 2
            and calcular_caracteres(zona_interes["edad"]) <= 3,
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
            "caracteres": calcular_caracteres(zona_interes["pregunta_1_si"])
            + calcular_caracteres(zona_interes["pregunta_1_no"])
            == 1,
            "palabras": True,
        },
        "pregunta_2": {
            "caracteres": calcular_caracteres(zona_interes["pregunta_2_si"])
            + calcular_caracteres(zona_interes["pregunta_2_no"])
            == 1,
            "palabras": True,
        },
        "pregunta_3": {
            "caracteres": calcular_caracteres(zona_interes["pregunta_3_si"])
            + calcular_caracteres(zona_interes["pregunta_3_no"])
            == 1,
            "palabras": True,
        },
        "comentarios": {
            "caracteres": calcular_caracteres(zona_interes["comentarios"]) <= 25,
            "palabras": calcular_caracteres(zona_interes["comentarios"])
            >= 1,  # Al menos un carácter determina una unica palabra
        },
    }

    for _, resultados in resultado_validaciones.items():
        # Guardar un booleano para mantener el tipo homogéneo en el diccionario
        resultados["es_valido"] = all(resultados.values())

    return (
        resultado_validaciones,
        all(resultados["es_valido"] for resultados in resultado_validaciones.values()),
    )


def main():
    """Función principal que procesa todos los formularios y genera resultados."""

    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

    imagenes_path = BASE_DIR / "imagenes"
    formularios = sorted(
        p
        for p in imagenes_path.glob("formulario_*.png")
        if p.stem != "formulario_vacio"
    )

    imagenes = []

    for formulario in formularios:
        imagenes.append(cv2.imread(str(formulario), cv2.IMREAD_GRAYSCALE))

    encabezados = [
        "id",
        "nombre y apellido",
        "edad",
        "mail",
        "legajo",
        "pregunta 1",
        "pregunta 2",
        "pregunta 3",
        "comentarios",
    ]
    filas = []
    canvases = []

    # Punto A y B
    for i, img in enumerate(imagenes):
        resultado_validaciones, es_valido = analisis_formulario(img, False)

        print("--------------------------------\n")
        print(f"Formulario {formularios[i].name}:")
        for campo, resultados in resultado_validaciones.items():
            resultado = "OK" if resultados["es_valido"] else "MAL"
            print(f"  - {campo}: {resultado}")

        print("\n--------------------------------\n")

        img_thresh = generar_imagen_binarizada(img, False)
        formulario = buscar_coordenadas_formulario(img_thresh, False)
        zona_interes = generar_roi(formulario, img_thresh, False)

        _, _, _, centroids = cv2.connectedComponentsWithStats(
            zona_interes["nombre_apellido"], connectivity=8, ltype=cv2.CV_32S
        )

        marca = (
            dibujar_check(zona_interes["nombre_apellido"], _centro_valido(centroids))
            if es_valido
            else dibujar_x(zona_interes["nombre_apellido"], _centro_valido(centroids))
        )

        canvases.append(marca)

        filas.append(
            [
                formularios[i].stem,
                (
                    "OK"
                    if resultado_validaciones["nombre_apellido"]["es_valido"]
                    else "MAL"
                ),
                "OK" if resultado_validaciones["edad"]["es_valido"] else "MAL",
                "OK" if resultado_validaciones["mail"]["es_valido"] else "MAL",
                "OK" if resultado_validaciones["legajo"]["es_valido"] else "MAL",
                "OK" if resultado_validaciones["pregunta_1"]["es_valido"] else "MAL",
                "OK" if resultado_validaciones["pregunta_2"]["es_valido"] else "MAL",
                "OK" if resultado_validaciones["pregunta_3"]["es_valido"] else "MAL",
                "OK" if resultado_validaciones["comentarios"]["es_valido"] else "MAL",
            ]
        )

    # Visualizar el plot con los resultados - Punto C
    n_formularios = len(canvases)
    n_rows = n_formularios
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols, constrained_layout=True)
    fig.suptitle("Resumen de validaciones", fontsize=16)

    ejes = axes.flatten()

    for idx, (canvas, formulario_path) in enumerate(zip(canvases, formularios)):
        ejes[idx].imshow(canvas, cmap="gray", vmin=0, vmax=255)
        ejes[idx].set_title(formulario_path.stem, fontsize=12)
        ejes[idx].axis("off")

    plt.show(block=False)

    # Guardar la imagen del plot - Punto C
    imagen_path = BASE_DIR / "resultados_formularios.png"
    fig.savefig(imagen_path)
    print(f"\nImagen guardada en: {imagen_path}")

    # Punto D
    csv_path = BASE_DIR / "resultados_formularios.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(encabezados)
        writer.writerows(filas)

    print(f"\nCSV guardado en: {csv_path}")


if __name__ == "__main__":
    main()
