# data_loader.py
from scipy.io import loadmat
from pathlib import Path
import pandas as pd
import json
from models import Car

# Rutas relativas basadas en la ubicación del script
BASE_DIR   = Path(__file__).resolve().parent         # …/backCars/BackFlask-asf-main
BACKCARS   = BASE_DIR.parent                         # …/backCars
PROJ_ROOT  = BACKCARS.parent                         # …/Proyecto computacion 2
MISC_DIR   = PROJ_ROOT / 'data' / 'misc'

# Archivos de atributos, modelo y mapeo
ATTR_PATH = MISC_DIR / 'attributes.txt'
MAT_PATH  = MISC_DIR / 'make_model_name.mat'
MAP_PATH  = MISC_DIR / 'model_make_map.json'

# Columnas del archivo attributes.txt
COLS = [
    'model_id', 'max_speed', 'displacement', 'doors', 'seats',
    'type_id', 'transmission', 'fuel_type', 'horsepower', 'price'
]

def load_all_cars():
    # 1) Leer atributos (omitir cabecera)
    df = pd.read_csv(
        str(ATTR_PATH), sep=r"\s+", names=COLS,
        skiprows=1, header=None
    )

    # 2) Leer marcas y modelos desde el .mat (ahora completo)
    mat    = loadmat(str(MAT_PATH), squeeze_me=True)
    makes  = [str(x).strip() for x in mat['make_names']]
    models = [str(x).strip() for x in mat['model_names']]

    # 3) Cargar mapeo de model_id a make_id
    with open(str(MAP_PATH), 'r', encoding='utf-8') as f:
        model_make_map = json.load(f)

    # 4) Filtrar solo model_id válidos (entre 1 y len(models))
    df = df[df['model_id'].between(1, len(models))]

    cars = []
    for _, row in df.iterrows():
        model_id = int(row['model_id'])
        make_id  = int(model_make_map.get(str(model_id), 0))

        # Obtener nombre real de la marca y del modelo
        brand      = makes[make_id-1] if 1 <= make_id <= len(makes) else 'Unknown'
        model_name = models[model_id-1] if 1 <= model_id <= len(models) else ''

        car = Car(
            model_id    = model_id,
            name        = f"{brand} {model_name}",
            velocity    = f"{int(row['max_speed'])} km/h",
            image_front = '',   # se completará en el endpoint
            image_rear  = '',
            fuel_type   = row['fuel_type'],
            horsepower  = row['horsepower'],
            price       = float(row['price']),
            passengers  = int(row['seats']),
            brand       = brand
        )
        cars.append(car)

    return cars
