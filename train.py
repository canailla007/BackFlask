# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1) Ruta a tu fichero de atributos (ya renombrado a attributes.txt)
ATTR_PATH = r'C:\Users\Usuario\Desktop\Ingenieria\Proyecto computacion 2\data\misc\attributes.txt'

# 2) Columnas de ese TXT (incluyendo transmission, fuel_type, horsepower y price)
COLS = [
    'model_id',
    'max_speed',
    'displacement',
    'doors',
    'seats',
    'type_id',
    'transmission',
    'fuel_type',
    'horsepower',
    'price'
]

# 3) Leer el archivo, saltando la primera línea de encabezado
df = pd.read_csv(
    ATTR_PATH,
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
        'fuel_type': str,
        'horsepower': str,  # lo dejamos como texto
        'price': float
    }
)

# 4) Convertir horsepower a número (quitar ' HP' y pasar a int)
df['hp_value'] = df['horsepower'].str.replace(' HP', '', regex=False).astype(int)

# 5) Crear la etiqueta is_family (ajusta los type_id que consideres «familiares»)
FAMILY_TYPES = [4, 7, 8]
df['is_family'] = df['type_id'].apply(lambda x: 1 if x in FAMILY_TYPES else 0)

# 6) Seleccionar las columnas numéricas para entrenar
X = df[['max_speed', 'displacement', 'doors', 'seats', 'price', 'hp_value']]
y = df['is_family']

# 7) Separar en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 8) Entrenar el modelo
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 9) Evaluar
print("Accuracy:", clf.score(X_test, y_test))

# 10) Guardar
joblib.dump(clf, 'family_model.pkl')