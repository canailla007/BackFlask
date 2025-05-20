# build_model_make_map.py

import json
from pathlib import Path

def main():
    # 1) Localiza la carpeta raíz del proyecto y la carpeta de imágenes
    BASE_DIR   = Path(__file__).resolve().parent            # …/backCars/BackFlask-asf-main
    PROJ_ROOT  = BASE_DIR.parent.parent                      # …/Proyecto computacion 2
    IMAGE_ROOT = PROJ_ROOT / 'data' / 'image'                # …/data/image
    OUTPUT     = PROJ_ROOT / 'data' / 'misc' / 'model_make_map.json'

    # 2) Recorre image/<make_id>/<model_id>/…
    model_to_make = {}
    for make_dir in IMAGE_ROOT.iterdir():
        if not make_dir.is_dir():
            continue
        make_id = make_dir.name
        for model_dir in make_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_id = model_dir.name
            # mapea model_id → make_id
            model_to_make[int(model_id)] = int(make_id)

    # 3) Guarda el diccionario en un JSON
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(model_to_make, f, indent=2, ensure_ascii=False)

    print(f"Guardado map model→make en: {OUTPUT}")

if __name__ == '__main__':
    main()
