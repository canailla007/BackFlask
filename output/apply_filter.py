#!/usr/bin/env python3
import shutil
from pathlib import Path

# Ajusta esta ruta si es necesario
DROP_TXT = Path(r"C:\Users\Usuario\Desktop\Ingenieria\Proyecto computacion 2\backCars\BackFlask-asf-main\output\dropped_folders.txt")

if not DROP_TXT.exists():
    print(f"No se encontró {DROP_TXT}")
    exit(1)

with DROP_TXT.open('r', encoding='utf-8') as f:
    for line in f:
        folder = Path(line.strip())
        if folder.exists() and folder.is_dir():
            try:
                shutil.rmtree(folder)
                print(f"Borrada carpeta: {folder}")
            except Exception as e:
                print(f"Error borrando {folder}: {e}")
        else:
            print(f"No existe o no es carpeta: {folder}")

print("Proceso completado.")
