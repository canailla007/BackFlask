#!/usr/bin/env python3
"""
train_improved.py: Entrenamiento avanzado de RandomForest con pipeline y GridSearchCV
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Rutas relativas basadas en la ubicación del script
BASE_DIR   = Path(__file__).resolve().parent         # …/backCars/BackFlask-asf-main
BACKCARS   = BASE_DIR.parent                         # …/backCars
PROJ_ROOT  = BACKCARS.parent                         # …/Proyecto computacion 2
MISC_DIR   = PROJ_ROOT / 'data' / 'misc'

# Archivos de atributos, modelo y mapeo
ATTR_PATH = MISC_DIR / 'attributes.txt'

COLS = [
    'model_id', 'max_speed', 'displacement', 'doors', 'seats',
    'type_id', 'transmission', 'fuel_type', 'horsepower', 'price'
]

# 2) Cargar datos
df = pd.read_csv(
    ATTR_PATH,
    sep=r"\s+",
    names=COLS,
    skiprows=1,
    header=None,
    dtype={
        'max_speed': float,
        'displacement': float,
        'doors': int,
        'seats': int,
        'type_id': int,
        'transmission': str,
        'fuel_type': str,
        'horsepower': str,
        'price': float
    }
)

# 3) Procesar horsepower a numérico
df['hp_value'] = df['horsepower'].str.replace(' HP', '', regex=False).astype(int)

# 4) Definir etiqueta is_family
FAMILY_TYPES = [4, 7, 8]  # ajustar según conveniencia
df['is_family'] = df['type_id'].isin(FAMILY_TYPES).astype(int)

# 5) Preparar X e y
feature_cols = ['max_speed', 'displacement', 'doors', 'seats', 'price', 'hp_value', 'transmission', 'fuel_type']
X = df[feature_cols]
y = df['is_family']

# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Pipeline y ColumnTransformer
numeric_features = ['max_speed', 'displacement', 'doors', 'seats', 'price', 'hp_value']
categorical_features = ['transmission', 'fuel_type']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# 8) GridSearchCV
param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

print("Iniciando búsqueda de hiperparámetros...")
grid.fit(X_train, y_train)
print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor AUC (CV): {grid.best_score_:.4f}")

# 9) Evaluación en test
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print("\n--- Evaluación en conjunto de test ---")
print(classification_report(y_test, y_pred))
print(f"ROC AUC (test): {roc_auc_score(y_test, y_proba):.4f}")
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# 10) Guardar modelo final
output_path = BASE_DIR / 'family_model.pkl'
joblib.dump(best_model, output_path)
print(f"Modelo guardado en: {output_path}")
