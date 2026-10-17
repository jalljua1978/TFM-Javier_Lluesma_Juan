"""Predictores reproducibles con validación por fórmula y descriptores Magpie.

Las predicciones proceden solo de la composición química: nunca sustituyen una
medición, ni el cálculo de DFT para un cristal particular.
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Mantengo una semilla fija para que la selección y el entrenamiento
# sean comparables al repetir el procedimiento.
SEED = 42


# Comparo cuatro enfoques, no doy por bueno un modelo sin evaluarlo.
# Los transformadores de datos viven dentro de cada pipeline.
def estimadores():
    return {
        'KNN': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), KNeighborsRegressor(n_neighbors=3)),
        'Random Forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=120, random_state=SEED, n_jobs=-1)),
        'Gradient Boosting': make_pipeline(SimpleImputer(strategy='median'), GradientBoostingRegressor(random_state=SEED)),
        'MLP': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), MLPRegressor(hidden_layer_sizes=(40, 20), max_iter=450, early_stopping=True, random_state=SEED))
    }


# Magpie traduce una fórmula química a números que entiende el modelo.
# Aquí NO usamos el resultado que queremos predecir como entrada.
def descriptores_composicion(formulas):
    """Magpie de composition únicamente. No emplear target/ID/descriptores derivados del target."""
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty
    # Los descriptores se calculan igual para MP y para los candidatos generados.
    f = ElementProperty.from_preset('magpie')
    rows = []
    # Si una fórmula no se interpreta, marcamos sus descriptores como ausentes
    # para que el análisis decida si puede continuar.
    for formula in formulas:
        try:
            c = Composition(str(formula))
            values = f.featurize(c)
            rows.append(values)
        except Exception:
            rows.append([float('nan')] * len(f.feature_labels()))
    vals = pd.DataFrame(rows, columns=f.feature_labels())
    return vals.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


# Antes de entrenar quito IDs duplicados y filas sin objetivo conocido.
# El recorte opcional de filas se deja documentado con una semilla.
def preparar_entrenamiento(df, target, max_rows=0):
    from pymatgen.core import Composition
    x = df[['material_id', 'formula_pretty', target]].copy()
    x[target] = pd.to_numeric(x[target], errors='coerce')
    x = x.dropna(subset=['formula_pretty', target]).drop_duplicates('material_id')
    if target == 'energy_above_hull': x = x[x[target] >= 0]
    if target == 'band_gap': x = x[x[target] >= 0]
    x = x.sort_values('material_id').reset_index(drop=True)
    # Este muestreo evita cargas excesivas, pero reduce el conjunto:
    # por eso se debe mencionar en la metodología.
    if max_rows and len(x) > max_rows:
        # Muestreo reproducible; para el estudio final debe documentarse el recorte.
        x = x.sample(n=int(max_rows), random_state=SEED).sort_values('material_id').reset_index(drop=True)
    if len(x) < 30: raise ValueError(f'No hay suficientes registros válidos para {target} ({len(x)} < 30)')
    try:
        groups=x['formula_pretty'].map(lambda s: Composition(str(s)).reduced_formula).astype(str)
    except Exception as exc:
        raise ValueError('Hay fórmulas no válidas en la base de entrenamiento') from exc
    return x, groups


# La evaluación se realiza por fórmulas distintas con GroupKFold.
# Esta separación es más exigente que dividir filas al azar sin mirar duplicados.
def entrenar_target(df, target, out, num_folds=5, max_rows=0):
    work, groups = preparar_entrenamiento(df, target, max_rows)
    if groups.nunique() < num_folds: raise ValueError('Número insuficiente de fórmulas distintas para validación agrupada')
    # Primero obtengo las variables y después descarto columnas vacías.
    features = descriptores_composicion(work.formula_pretty)
    valid_cols = list(features.columns[features.notna().any(axis=0)])
    if len(valid_cols) < 2: raise ValueError('No hay suficientes descriptores Magpie disponibles')
    X = features[valid_cols]
    y = work[target].to_numpy(dtype=float)
    cv = GroupKFold(n_splits=num_folds)
    evaluations=[]
    # MAE, RMSE y R² se calculan sobre pliegues no vistos durante el ajuste.
    # Después podré comparar modelos en la Tabla 14.
    for name, model in estimadores().items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res=cross_validate(model, X, y, cv=cv, groups=groups, scoring={
                'MAE':'neg_mean_absolute_error','RMSE':'neg_root_mean_squared_error','R2':'r2'},n_jobs=1,error_score='raise')
        evaluations.append(dict(variable_objetivo=target, modelo=name,
                                MAE=float(-np.mean(res['test_MAE'])),
                                RMSE=float(-np.mean(res['test_RMSE'])),
                                R2=float(np.mean(res['test_R2'])),
                                validacion='GroupKFold por formula reducida',
                                filas_entrenamiento=len(y),grupos=groups.nunique()))
    table=pd.DataFrame(evaluations).sort_values('RMSE').reset_index(drop=True)
    # Elijo el modelo con menor RMSE medio; es una regla explícita,
    # no el que casualmente produjo la figura más bonita.
    champion=str(table.iloc[0].modelo)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        oof=cross_val_predict(clone(estimadores()[champion]), X, y, groups=groups,cv=cv,n_jobs=1)
    predictions=pd.DataFrame(dict(material_id=work.material_id, formula_pretty=work.formula_pretty,
                                  target=target,real=y,predicho_oof=oof))
    # Solo después de validar entreno el modelo final con todos los datos.
    # Lo guardo para estimar las propiedades de los CIF nuevos.
    fitted=clone(estimadores()[champion]).fit(X,y)
    joblib.dump({'modelo':fitted,'features':valid_cols,'target':target,
                 'metodo':'Magpie composicion sola','model_version':'TFM_HITO3_2026_1'},out/f'modelo_{target}.joblib')
    predictions.to_csv(out/f'predicciones_oof_{target}.csv',index=False)
    table.to_csv(out/f'comparacion_modelos_{target}.csv',index=False)
    return table, predictions, features, work, valid_cols


# La misma transformación Magpie se aplica a los nuevos candidatos.
# Si alguna fórmula no tiene descriptores suficientes, lo notifico.
def predecir_modelos(formulas, resultados):
    formulas=list(formulas)
    X=descriptores_composicion(formulas)
    output={}
    for target in ('energy_above_hull','band_gap'):
        job=joblib.load(Path(resultados)/f'modelo_{target}.joblib')
        xx=X.reindex(columns=job['features'])
        if xx.isna().all(axis=1).any():
            bad=np.flatnonzero(xx.isna().all(axis=1))
            raise ValueError(f'Formulas sin descriptores: indices {bad.tolist()}')
        output[target]=job['modelo'].predict(xx)
    return output
