import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, abort, request
import random, glob, os, joblib
from data_loader import load_all_cars

# --- Rutas base usando pathlib ---
BASE_DIR    = Path(__file__).resolve().parent         # …/BackFlask-asf-main
PROJ_ROOT   = BASE_DIR.parent.parent                   # …/Proyecto computacion 2
DATA_DIR    = PROJ_ROOT / 'data'
IMAGE_ROOT  = DATA_DIR / 'image'
LABEL_ROOT  = DATA_DIR / 'label'
MAP_PATH    = PROJ_ROOT / 'data' / 'misc' / 'model_make_map.json'

# --- Carga mapa model→make ---
with open(MAP_PATH, 'r', encoding='utf-8') as f:
    model_make_map = json.load(f)

app   = Flask(__name__)
cars  = load_all_cars()
model = joblib.load('family_model.pkl')

def find_images(model_id):
    # 1) Busca make_id en el mapa
    make_id = model_make_map.get(str(model_id)) or model_make_map.get(model_id)
    if make_id is None:
        return [], []

    # 2) Construye patrón estricto
    pattern = str(
        LABEL_ROOT / str(make_id) / str(model_id) / '*' / '*.txt'
    )
    fronts, rears = [], []
    for lbl in glob.glob(pattern):
        view = Path(lbl).read_text(encoding='utf-8').splitlines()[0].strip()
        stem = Path(lbl).stem
        img  = IMAGE_ROOT / str(make_id) / str(model_id) / Path(lbl).parent.name / f"{stem}.jpg"
        if not img.exists():
            continue
        if view == '1':
            fronts.append(str(img))
        elif view == '2':
            rears.append(str(img))
    return fronts[:1], rears[:1]

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
    ), 200

@app.route('/cars', methods=['GET'])
def get_cars():
    sample = random.sample(cars, 10)
    out = []
    for car in sample:
        fronts, rears = find_images(car.model_id)
        d = car.to_dict()
        if fronts:
            rel = Path(fronts[0]).relative_to(IMAGE_ROOT).as_posix()
            d['image_front'] = f'/images/{rel}'
        if rears:
            rel = Path(rears[0]).relative_to(IMAGE_ROOT).as_posix()
            d['image_rear'] = f'/images/{rel}'
        out.append(d)
    return jsonify(out)

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
    data = request.json or {}
    hp = int(data.get('horsepower', '0').split()[0])
    vals = [
        data.get('max_speed', 0),
        data.get('displacement', 0),
        data.get('doors', 0),
        data.get('seats', 0),
        data.get('price', 0),
        hp
    ]
    proba = model.predict_proba([vals])[0][1]   # probabilidad de clase «1» (familia)
    is_fam = proba >= 0.5                       # umbral por defecto
    return jsonify({
        'is_family': bool(is_fam),
        'probability_family': round(float(proba), 3)
    })

if __name__ == '__main__':
    app.run(debug=True)