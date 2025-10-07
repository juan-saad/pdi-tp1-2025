# pdi-tp1-2025 — Instalación rápida

Requisitos: Python 3.10+ y conexión a Internet.

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
python -c "import numpy, matplotlib, cv2, PIL; print('Imports OK')"
```

1. Salir del venv

```powershell
deactivate
```

Notas

- Si `opencv-contrib-python` falla en Windows, instala Visual C++ Build Tools o usa `opencv-python-headless`.
- Para fijar versiones o declarar la versión de Python, puedo añadir `requirements.txt` con pins o un `pyproject.toml`.
