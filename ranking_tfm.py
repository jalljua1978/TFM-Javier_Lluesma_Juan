"""Evaluación multicriterio HONESTA del electrolito.

Dos productos diferentes:
- ranking_exploratorio: estabilidad y band gap predichos, y novedad de fórmula;
  no certifica propiedades funcionales de electrolitos.
- ranking_integral: solo con las SEIS propiedades documentadas con fuente y
  con pesos aprobados por quien realiza el TFM.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Estos son los criterios que motivan la investigación.
# No todos tienen un dato calculado: distinguirlo es clave en el TFM.
CRITERIOS_TFM = [
    ('estabilidad_termodinamica','Baja energía sobre la envolvente (eV/átomo)','Filtro primario', 'Energía estimada por ML/comparación MP'),
    ('conductividad_ionica','Alta conductividad de Li+ (S/cm)','Criterio principal','NO calculada por MatterGen ni por MP summary'),
    ('estabilidad_electroquimica','Ventana electroquímica amplia (V)','Criterio principal','NO calculada por band gap'),
    ('compatibilidad_electrodos','Compatibilidad química/interfacial','Complementario','Requiere cálculos/experimentos de interfaces'),
    ('propiedades_mecanicas','Comportamiento mecánico adecuado','Complementario','Requiere evaluaciones mecánicas'),
    ('novedad','Diversidad y ausencia de duplicados','Control de generación','Novedad de fórmula no demuestra novedad estructural'),
]

# La matriz deja explícito qué propiedades podemos evaluar de verdad.
def guardar_matriz_criterios(out: Path):
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'criterio':a,'objetivo':b,'funcion_tfm':c,'estado_de_dato':d} for a,b,c,d in CRITERIOS_TFM]).to_csv(out/'tabla_11_criterios_tfm.csv',index=False)


# Como la energía y el band gap tienen unidades diferentes, paso
# a una escala común 0-1 antes de sumarlos.
def normalizar(serie, sentido):
    x=pd.to_numeric(serie,errors='coerce')
    if x.isna().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError(f'Criterio con valores no numéricos/ausentes: {serie.name}')
    if sentido not in ('min','max'):
        raise ValueError('sentido debe ser min o max')
    rng=float(x.max()-x.min())
    # Si todos sacan lo mismo en un criterio, doy 0,5 a todos.
    # Así esa propiedad no favorece arbitrariamente a nadie.
    if np.isclose(rng,0):
        # Una propiedad constante no discrimina; asignar 0.5 a todos y documentarlo.
        return pd.Series(0.5,index=x.index,dtype=float)
    val=(x-x.min())/rng
    return (1-val if sentido=='min' else val).astype(float)


# La puntuación suma las propiedades normalizadas según los pesos.
# Compruebo que sumen 1 para que el ranking tenga sentido.
def clasificar(df, criterios, id_col='material_id'):
    if df.empty: raise ValueError('No se puede clasificar un conjunto vacío')
    if id_col not in df: raise ValueError(f'Falta ID {id_col}')
    if df[id_col].duplicated().any(): raise ValueError('Candidatos con IDs duplicados')
    if len(criterios)<2: raise ValueError('La evaluación multicriterio requiere >=2 criterios')
    pesos=[float(c['peso']) for c in criterios]
    if any(w<0 or not np.isfinite(w) for w in pesos) or not np.isclose(sum(pesos),1,atol=1e-8):
        raise ValueError('Los pesos deben ser no negativos y sumar 1')
    ranked=df.copy().reset_index(drop=True)
    columnas=[]
    # Cada criterio puede premiar el máximo o el mínimo;
    # por ejemplo, en la energía sobre hull interesa un valor menor.
    for c in criterios:
        columna=c['columna']
        if columna not in ranked: raise ValueError('Falta columna de criterio: '+columna)
        dst='norm_'+columna
        ranked[dst]=normalizar(ranked[columna],c['sentido'])
        columnas.append(dst)
    # Ordeno de mayor a menor y uso el ID como desempate reproducible.
    ranked['puntuacion']=ranked[columnas].to_numpy()@np.asarray(pesos)
    ranked=ranked.sort_values(['puntuacion',id_col],ascending=[False,True],kind='mergesort').reset_index(drop=True)
    ranked.insert(0,'posicion',np.arange(1,len(ranked)+1))
    return ranked, columnas


# Cambio uno por uno los pesos para ver cuánto se mueve la clasificación.
# Si el ranking cambia mucho, debemos reconocer esa limitación.
def sensibilidad(ranked, criterios, cols, out: Path, etiqueta: str):
    ids=ranked.material_id.astype(str).tolist()
    base=np.array([float(c['peso']) for c in criterios])
    scenarios={'Base':base}
    # Aquí doy un 50 % más de peso a cada propiedad de forma individual
    # y renormalizo para seguir sumando 1.
    for idx,c in enumerate(criterios):
        v=base.copy()
        v[idx]*=1.5
        v=v/v.sum()
        scenarios['Priorizar '+c['columna']]=v
    data={'material_id':ids}
    # Guardo la posición de cada candidato bajo cada escenario,
    # no solo una imagen: así se puede auditar la sensibilidad.
    for k,w in scenarios.items():
        score=ranked[cols].to_numpy()@w
        order=np.argsort(-score,kind='stable')
        ranks=np.empty(len(order),dtype=int);ranks[order]=np.arange(1,len(order)+1)
        data[k]=ranks
    sens=pd.DataFrame(data)
    sens.to_csv(out/f'analisis_sensibilidad_{etiqueta}.csv',index=False)
    top=min(5,len(ids))
    fig,ax=plt.subplots(figsize=(9,5.3))
    for i in range(top): ax.plot(list(scenarios),[sens[s].iloc[i] for s in scenarios],marker='o',label=ids[i])
    ax.invert_yaxis(); ax.set_ylabel('Posición (1 = mejor)');ax.set_title(f'Figura 30. Sensibilidad de ranking {etiqueta}')
    ax.tick_params(axis='x',rotation=30)
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'figura_30_sensibilidad_{etiqueta}.png',dpi=160);plt.close(fig)


# Este ranking usa tres indicadores disponibles. No lo llamamos
# ranking completo de electrolitos porque faltan propiedades esenciales.
def ranking_exploratorio(candidatos,config:dict,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    scfg=config['seleccion']
    criteria=[
        dict(columna='energy_above_hull_predicha_eV_atom',sentido='min',peso=float(scfg['pesos_exploratorios']['energy_above_hull_predicha'])),
        dict(columna='band_gap_predicho_eV',sentido='max',peso=float(scfg['pesos_exploratorios']['band_gap_predicho_proxy'])),
        dict(columna='novedad_formula',sentido='max',peso=float(scfg['pesos_exploratorios']['novedad_formula']))
    ]
    df=candidatos.copy()
    # Los filtros opcionales quedan en la configuración.
    # Si ninguno los supera, no invento una clasificación vacía.
    if scfg.get('max_energy_above_hull_predicha_ev_atom') is not None:
        df=df[df['energy_above_hull_predicha_eV_atom']<=float(scfg['max_energy_above_hull_predicha_ev_atom'])]
    if scfg.get('min_band_gap_ev') is not None:
        df=df[df['band_gap_predicho_eV']>=float(scfg['min_band_gap_ev'])]
    if df.empty: raise ValueError('Ningún candidato supera filtros; no se crea ranking')
    # Estos pesos (0,65 / 0,25 / 0,10) son exploratorios
    # y hay que justificarlos, no interpretarlos como una verdad física.
    ranked, cols=clasificar(df,criteria)
    ranked['tipo_ranking']='EXPLORATORIO — NO APTITUD COMO ELECTROLITO DEMOSTRADA'
    ranked.to_csv(out/'tabla_16_ranking_EXPLORATORIO.csv',index=False)
    pd.DataFrame([{**c,'etiqueta':'Solo exploratorio','justificacion':'La conductividad iónica y la ventana electroquímica no se han medido ni calculado'} for c in criteria]).to_csv(out/'tabla_15_pesos_EXPLORATORIOS.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    top=ranked.head(5)
    top.set_index('material_id')[cols].plot(kind='bar',ax=ax,ylim=(0,1))
    ax.set_ylabel('Valor normalizado');ax.set_title('Figura 29. Indicadores exploratorios (no propiedades funcionales completas)')
    ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(out/'figura_29_propiedades_exploratorias.png',dpi=160);plt.close(fig)
    sensibilidad(ranked,criteria,cols,out,'exploratorio')
    return ranked


COLUMNAS_COMPLETAS=[
    ('energy_above_hull_validada_eV_atom','min'),
    ('conductividad_ionica_S_cm','max'),
    ('ventana_electroquimica_V','max'),
    ('compatibilidad_interfacial_puntuacion_0_1','max'),
    ('propiedades_mecanicas_puntuacion_0_1','max'),
    ('novedad_formula','max'),
]

# He previsto un ranking integral solo por si llegamos a tener
# validaciones adicionales con metodología y referencia verificables.
def ranking_integral(candidatos, archivo_validaciones, pesos, out):
    """Solo datos aportados por un estudio con fuente verificable. No inventar.

    Requiere columnas de procedencia por material: metodo, referencia.
    La energía puede ser estimación por DFT/MLFF, documentada en procedencia.
    """
    if not archivo_validaciones: return False
    inp=Path(archivo_validaciones)
    if not inp.is_file(): raise FileNotFoundError(inp)
    # El fichero externo debe identificar material, método y fuente.
    # Sin esa trazabilidad, no le daría valor como evidencia en la defensa.
    v=pd.read_csv(inp)
    required=['material_id','metodo','referencia']+[k for k,_ in COLUMNAS_COMPLETAS if k!='novedad_formula']
    missing=[c for c in required if c not in v]
    if missing: raise ValueError('Validaciones científicas incompletas; faltan: '+', '.join(missing))
    if v.material_id.duplicated().any(): raise ValueError('Hay duplicados en datos validados')
    if v['metodo'].isna().any() or v['referencia'].isna().any() or (v['referencia'].astype(str).str.strip()=='').any():
        raise ValueError('Cada validación requiere metodología y referencia')
    if not pesos or set(pesos)!={k for k,_ in COLUMNAS_COMPLETAS}:
        raise ValueError('Pesos del ranking integral deben definirse explícitamente para los seis criterios')
    # Los cuatro campos funcionales se inicializan en NaN en el fichero de
    # trazabilidad; al incorporar validaciones externas se sustituyen, no
    # se mezclan creando columnas _x/_y.
    cols_aportadas=[k for k,_ in COLUMNAS_COMPLETAS if k!='novedad_formula']
    tabla_base=candidatos.drop(columns=cols_aportadas,errors='ignore')
    # Uno los datos por ID para evitar asignar propiedades al cristal equivocado.
    df=tabla_base.merge(v,on='material_id',how='inner',validate='one_to_one')
    if df.empty: raise ValueError('No hay coincidencias entre IDs de candidatos y validaciones')
    # No reemplazamos sin aviso ausencias por valores inventados.
    # Si falta una medida, se detiene: no relleno huecos con cero
    # porque eso alteraría artificialmente el resultado.
    required_numeric=[k for k,_ in COLUMNAS_COMPLETAS]
    if df[required_numeric].isna().any().any(): raise ValueError('Hay validaciones numéricas ausentes para algunos candidatos')
    for c in required_numeric:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    if df[required_numeric].isna().any().any() or not np.isfinite(df[required_numeric].to_numpy(dtype=float)).all():
        raise ValueError('Las validaciones deben contener números finitos en todos los criterios')
    if (df['conductividad_ionica_S_cm']<=0).any(): raise ValueError('La conductividad debe ser > 0 S/cm')
    if (df['energy_above_hull_validada_eV_atom']<0).any(): raise ValueError('Energy above hull no puede ser negativa')
    if (df['ventana_electroquimica_V']<0).any(): raise ValueError('Ventana electroquímica no puede ser negativa')
    for k in ('compatibilidad_interfacial_puntuacion_0_1','propiedades_mecanicas_puntuacion_0_1'):
        if not pd.to_numeric(df[k]).between(0,1).all(): raise ValueError(f'{k}: exige puntuación definida en [0,1]')
    criteria=[{'columna':k,'sentido':sentido,'peso':pesos[k]} for k,sentido in COLUMNAS_COMPLETAS]
    ranked, cols=clasificar(df,criteria)
    ranked['tipo_ranking']='INTEGRAL CON DATOS APORTADOS; SEGUIR LIMITACIONES METODOLÓGICAS'
    ranked.to_csv(out/'tabla_16_ranking_integral_CON_DATOS.csv',index=False)
    pd.DataFrame(criteria).to_csv(out/'tabla_15_pesos_integrales.csv',index=False)
    sensibilidad(ranked,criteria,cols,out,'integral')
    return True
