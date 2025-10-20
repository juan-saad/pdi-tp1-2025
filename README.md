# Trabajo practico 1 - Procesamiento de imagenes

**Requisitos**: Python 3.12.10 y conexión a Internet.

## Instalación

1. Crear y activar venv (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

1. Actualizar pip e instalar dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

1. Verificar

```powershell
python -c "import numpy, matplotlib, cv2; print('Imports OK')"
```

## Ejecución

### Problema 2 - Análisis de Formularios

Para ejecutar el análisis de formularios:

```powershell
python problema2.py
```

ó

Seleccionar todas las lineas del programa y apretar la combinación de teclas `Shift + Enter`

**Este script**:

- Procesa todos los formularios en la carpeta `imagenes/` (excepto `formulario_vacio.png`)
- Valida cada campo según criterios específicos (caracteres, palabras)
- Genera visualizaciones de las regiones de interés y proyecciones
- Muestra un resumen con marcas de validación (✓ o ✗) en cada formulario
- Exporta los resultados a `resultados_formularios.csv` y a `resultados_formularios.png`

**Nota:** El script mostrará múltiples ventanas de visualización. Para evitar que se abran los plots, se definió un parametro boleano que controla si se muestran o no los mismos. Ejemplo:

```python
analisis_formulario(img, False) # Ejecuta el analisis sin abrir ningun plot
```

## Salir del venv

```powershell
deactivate
```
