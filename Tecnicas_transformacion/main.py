# app.py

import os
import sys

# Cambiar al directorio de Tecnicas_transformacion para que las importaciones relativas funcionen
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# Ahora importar la interfaz
from interfaz.interfaz import run_app

if __name__ == "__main__":
    run_app()
