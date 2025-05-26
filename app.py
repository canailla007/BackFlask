import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, abort, request, url_for
import random, glob, os, joblib

# ——— Nuevas importaciones para el recommender ———
import pandas as pd
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors

from data_loader import load_all_cars

# --- Rutas base usando pathlib ---
BASE_DIR    = Path(__file__).resolve().parent
PROJ_ROOT   = BASE_DIR.parent.parent
DATA_DIR    = PROJ_ROOT / 'data'
IMAGE_ROOT  = DATA_DIR / 'image'
LABEL_ROOT  = DATA_DIR / 'label'
MISC_DIR    = PROJ_ROOT / 'data' / 'misc'
MAP_PATH    = MISC_DIR / 'model_make_map.json'
ATTR_PATH   = MISC_DIR / 'attributes.txt'        # atributos :contentReference[oaicite:0]{index=0}
MAT_PATH    = MISC_DIR / 'make_model_name.mat'

# --- Carga mapa model→make ---
with open(MAP_PATH, 'r', encoding='utf-8') as f:
    model_make_map = json.load(f)

app   = Flask(__name__)
cars  = load_all_cars()
model = joblib.load('family_model.pkl')            # modelo familia

# ——— Setup de datos para el recommender ———
# 1) Leer atributos: max_speed, displacement, doors, seats, fuel_type, horsepower, price :contentReference[oaicite:1]{index=1}
df_attr = pd.read_csv(
    ATTR_PATH,
    sep=r"\s+",
    names=['model_id','max_speed','displacement','doors','seats','type_id','transmission','fuel_type','horsepower','price'],
    skiprows=1,
    header=None
)

# 2) Cargar nombres de modelos para validar model_id
mat = loadmat(str(MAT_PATH), squeeze_me=True)
n_models = len(mat['model_names'])
df_attr = df_attr[df_attr['model_id'].between(1, n_models)]

# 3) Limpiar horsepower y convertir a entero
df_attr['hp'] = df_attr['horsepower'].str.replace(' HP','', regex=False).astype(int)

# 4) Construir matriz de características y ajustar KNN (k=5)
feature_matrix = df_attr[['max_speed','displacement','doors','seats','price','hp']].values
knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(feature_matrix)

def find_images(model_id):
    make_id = model_make_map.get(str(model_id)) or model_make_map.get(model_id)
    if make_id is None:
        return [], []

    make_str  = str(make_id)
    model_str = str(model_id)

    base_img_dir = IMAGE_ROOT / make_str / model_str
    base_lbl_dir = LABEL_ROOT / make_str / model_str

    fronts, rears = [], []

    # Si no existe la carpeta de imágenes, salimos
    if not base_img_dir.exists():
        return fronts, rears

    # Recorremos por año de lanzamiento
    for year_dir in base_img_dir.iterdir():
        if not year_dir.is_dir():
            continue

        # Cada archivo .jpg
        for img_path in year_dir.glob('*.jpg'):
            # Ruta al .txt de etiqueta correspondiente
            lbl_path = base_lbl_dir / year_dir.name / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # Primera línea: viewpoint (1 frontal, 2 trasera)
            with open(lbl_path, 'r') as f:
                vp = f.readline().strip()

            # Añadimos sólo la primera frontal y la primera trasera
            if vp == '1' and not fronts:
                fronts.append(str(img_path))
            elif vp == '2' and not rears:
                rears.append(str(img_path))

            # Si ya tenemos ambos, salimos
            if fronts and rears:
                break
        if fronts and rears:
            break

    return fronts, rears


def _parse_request_features(data):
    """
    Extrae de `data` el vector de características [max_speed, displacement, doors, seats, price, hp]
    y el fuel_type (si viene).
    """
    data = data or {}
    # Parsear horsepower (puede venir "100 HP" o número)
    hp = int(str(data.get('horsepower', '0')).split()[0])
    feat_vec = [
        float(data.get('max_speed', 0)),
        float(data.get('displacement', 0)),
        int(data.get('doors', 0)),
        int(data.get('seats', 0)),
        float(data.get('price', 0)),
        hp
    ]
    fuel = data.get('fuel_type', None)
    return feat_vec, fuel

def _serialize_cars(car_list):
    """
    Dado un iterable de objetos Car, construye la lista de dicts
    añadiendo image_front e image_rear cuando existan.
    Las URLs de imagen se generan completas (incluyen esquema, host y puerto).
    """
    out = []
    for car in car_list:
        fronts, rears = find_images(car.model_id)
        d = car.to_dict()

        if fronts:
            rel = Path(fronts[0]).relative_to(IMAGE_ROOT).as_posix()
            # _external=True hace que devuelva "http://127.0.0.1:5000/images/..."
            d['image_front'] = url_for('serve_image', relpath=rel, _external=True)

        if rears:
            rel = Path(rears[0]).relative_to(IMAGE_ROOT).as_posix()
            d['image_rear'] = url_for('serve_image', relpath=rel, _external=True)

        out.append(d)
    return out

@app.route('/images/<path:relpath>')
def serve_image(relpath):
    return send_from_directory(str(IMAGE_ROOT), relpath)

@app.route('/', methods=['GET'])
def home():
    return (
        "API corriendo:\n"
        " - GET  /cars                   → 10 coches aleatorios con imágenes\n"
        " - GET  /cars/<model_id>/images → rutas frontal y trasera de un modelo\n"
        " - POST /predict-family         → comprueba si un coche es familiar\n"
        " - POST /recommender/get        → recomienda 5 coches similares\n"
    ), 200

@app.route('/cars', methods=['GET'])
def get_cars():
    sample = random.sample(cars, 10)
    return jsonify(_serialize_cars(sample))

@app.route('/cars/<int:model_id>/images', methods=['GET'])
def get_car_images(model_id):
    fronts, rears = find_images(model_id)
    if not fronts and not rears:
        abort(404, 'No images for this model_id')
    res = {}
    for name, imgs in (('image_front', fronts), ('image_rear', rears)):
        if imgs:
            rel = Path(imgs[0]).relative_to(IMAGE_ROOT).as_posix()
            res[name] = f'/images/{rel}'
        else:
            res[name] = ''
    return jsonify(res)

@app.route('/predict-family', methods=['POST'])
def predict_family():
    data = request.json

    # 1) Extraer hp_value
    hp_value = int(data['horsepower'].replace(' HP', ''))

    # 2) Crear un DataFrame con *todas* las columnas que entrenaste
    df_feat = pd.DataFrame([{
        'max_speed':    data['max_speed'],
        'displacement': data['displacement'],
        'doors':        data['doors'],
        'seats':        data['seats'],
        'price':        data['price'],
        'hp_value':     hp_value,
        'transmission': data['transmission'],
        'fuel_type':    data['fuel_type']
    }])

    # 3) Predecir con el pipeline completo
    proba = model.predict_proba(df_feat)[0][1]
    res = {
        'is_family':         bool(proba >= 0.5),
        'probability_family': round(float(proba), 3)
    }
    return jsonify(res)

@app.route('/recommender/get', methods=['POST'])
def get_recommendations():
    feat_vec, fuel = _parse_request_features(request.json)

    # Si hay fuel_type y al menos 5 coches de ese tipo, filtramos antes
    if fuel:
        subset = df_attr[df_attr['fuel_type'] == fuel]
        if len(subset) >= 5:
            feats = subset[['max_speed','displacement','doors','seats','price','hp']].values
            knn   = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(feats)
            _, idx = knn.kneighbors([feat_vec])
            rec_ids = subset.iloc[idx[0]]['model_id'].tolist()
        else:
            _, idx = knn_model.kneighbors([feat_vec])
            rec_ids  = df_attr.iloc[idx[0]]['model_id'].tolist()
    else:
        _, idx = knn_model.kneighbors([feat_vec])
        rec_ids  = df_attr.iloc[idx[0]]['model_id'].tolist()

    # Mapear a objetos Car, ordenados según rec_ids
    reordered = sorted(
        (c for c in cars if c.model_id in rec_ids),
        key=lambda c: rec_ids.index(c.model_id)
    )[:5]
    return jsonify(_serialize_cars(reordered))

if __name__ == '__main__':
    app.run(debug=True)
