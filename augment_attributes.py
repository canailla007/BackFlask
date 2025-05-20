import pandas as pd
import random

# --- Ajusta estas rutas a tu proyecto ---
INPUT_PATH  = r'C:\Users\Usuario\Desktop\Ingenieria\Proyecto computacion 2\data\misc\attributes_extendido.txt'
OUTPUT_PATH = r'C:\Users\Usuario\Desktop\Ingenieria\Proyecto computacion 2\data\misc\attributes_with_price.txt'

# Columnas originales de tu fichero
COLS = [
    'model_id',
    'max_speed',
    'displacement',
    'doors',
    'seats',
    'type_id',
    'transmission',
    'fuel'
]

def main():
    # 1) Leer el fichero, saltando la línea de cabecera
    df = pd.read_csv(
        INPUT_PATH,
        sep=r'\s+',
        names=COLS,
        skiprows=1,
        dtype={
            'model_id': int,
            'max_speed': float,
            'displacement': float,
            'doors': int,
            'seats': int,
            'type_id': int,
            'transmission': str,
            'fuel': str
        }
    )

    # 2) Generar campos aleatorios
    df['horsepower'] = df.apply(lambda _: f"{random.randint(100, 500)} HP", axis=1)
    df['price']      = df.apply(lambda _: round(random.uniform(20000, 200000), 2), axis=1)

    # 3) Volcar a un nuevo .txt (incluye encabezado para luego saltarlo con skiprows=1)
    df.to_csv(
        OUTPUT_PATH,
        sep=' ',
        index=False,
        header=True
    )
    print(f"Archivo generado: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()