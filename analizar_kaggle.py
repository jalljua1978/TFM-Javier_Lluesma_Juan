#!/usr/bin/env python3
"""Evaluación científica de CIF realmente generados; jamás inventa propiedades faltantes."""
from __future__ import annotations
import argparse, json, os, shutil, sys, traceback, zipfile
from collections import Counter, defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generar_kaggle import load_config, manifest, plan_batches, write_json, now
from modelos_tfm import entrenar_target, predecir_modelos
from graficos_tfm import grafico_20, figura_28, figuras_23_24
from ranking_tfm import ranking_exploratorio, guardar_matriz_criterios
from preparar_materiales import FIELDS, FAMILIES, family


# El conjunto de referencia viene de Materials Project, no de filas inventadas.
# Lo descargo una vez y lo guardo para poder revisar qué datos se usaron.
def get_reference(cfg, out):
    dest=out/'01_datos';dest.mkdir(parents=True,exist_ok=True)
    cleanpath=dest/'materials_limpios.csv'
    # Si ya tenemos datos descargados, evitamos repetir consultas a la API.
    # Esto ahorra tiempo y mantiene el mismo conjunto entre reintentos.
    if cleanpath.exists():
        clean=pd.read_csv(cleanpath)
        if len(clean)>30:
            raw=pd.read_csv(dest/'materials_descargados.csv')
            print(f'[DATOS] Reutilizando {len(clean)} entradas de MP guardadas',flush=True)
            return clean,len(raw),len(raw)-len(raw.drop_duplicates('material_id'))
    # La clave entra mediante una variable de entorno, desde los Secrets de Kaggle.
    # Nunca debe escribirse aquí ni guardarse en GitHub.
    key=os.environ.get('MP_API_KEY','').strip()
    if not key:
        raise RuntimeError('Falta MP_API_KEY. Crea el secreto MP_API_KEY en Kaggle antes del análisis; la clave no se guarda en el proyecto.')
    from mp_api.client import MPRester
    rows=[];records=[]
    # Consulto sistemas que contienen litio y los elementos elegidos.
    # La muestra de referencia no representa necesariamente toda Materials Project.
    with MPRester(key) as mpr:
        for elem in FAMILIES.values():
            print('[MP] Descargando',elem,flush=True)
            try:
                docs=list(mpr.materials.summary.search(elements=['Li',elem],fields=FIELDS))
            # Cuando un CIF no pasa, guardo el motivo del rechazo.
            # No borro silenciosamente los errores porque afectan al resultado final.
            except Exception as exc:
                records.append({'elemento':elem,'estado':'ERROR','mensaje':str(exc)[:150]})
                continue
            docs=sorted(docs,key=lambda d:str(d.material_id));take=int(cfg['datos']['max_por_consulta'])
            chosen=docs[:take] if take>0 else docs
            records.append({'elemento':elem,'estado':'OK','recuperados':len(docs),'incluidos':len(chosen)})
            for d in chosen:
                item={k:getattr(d,k,None) for k in FIELDS}
                item['material_id']=str(item['material_id'])
                elems=[str(x) for x in (item.get('elements') or [])]
                item['elements']=';'.join(elems);item['familias']=family(elems)
                rows.append(item)
    write_json(dest/'registro_descarga.json',{'fuente':'Materials Project (MP API), cálculos de referencia',
        'consultas':records,'fecha':now(),'muestreo':'primeros N identificadores ordenados, NO aleatorio'})
    if not rows: raise RuntimeError('MP no devolvió registros; revisar registro_descarga.json')
    # Primero guardo los datos tal como llegaron; luego limpio duplicados.
    # Así puedo explicar de dónde sale el número de la Tabla 12.
    raw=pd.DataFrame(rows);raw.to_csv(dest/'materials_descargados.csv',index=False)
    clean=raw.drop_duplicates('material_id').dropna(subset=['formula_pretty']).copy()
    for col in ['energy_above_hull','band_gap']:
        clean[col]=pd.to_numeric(clean[col],errors='coerce')
    # No entreno con energías ausentes o físicamente negativas.
    # Esta limpieza es sencilla, pero importa para no sesgar el modelo.
    clean=clean[clean.energy_above_hull.notna() & clean.energy_above_hull.ge(0)]
    clean=clean.sort_values('material_id').reset_index(drop=True)
    if len(clean)<60:raise RuntimeError(f'Solo {len(clean)} materiales de referencia; entrenamiento insuficiente')
    clean.to_csv(cleanpath,index=False)
    return clean,len(raw),len(raw)-len(raw.drop_duplicates('material_id'))


# Aquí sí revisamos los CIF de MatterGen. Los contamos, abrimos y filtramos.
# Generar un archivo no es lo mismo que demostrar que el material sirve.
def screening(cfg, root, out, reference):
    from pymatgen.core import Structure, Composition
    from pymatgen.analysis.structure_matcher import StructureMatcher
    data=[];reject=[];unique=defaultdict(list)
    # StructureMatcher ayuda a detectar estructuras muy parecidas.
    # Esto evita presentar dos cristales equivalentes como dos propuestas distintas.
    matcher=StructureMatcher(ltol=0.2,stol=0.3,angle_tol=5)
    # La novedad se compara solo con las fórmulas del subconjunto descargado.
    # No se puede afirmar que una composición es inédita en todo el mundo.
    known=set()
    for v in reference.formula_pretty.dropna().astype(str):
        try:known.add(Composition(v).reduced_formula)
        except Exception:pass
    source=Path(root)/'lotes';filenames=sorted(source.glob('*/generated_crystals_cif.zip'))
    allowed={entry['lote_id']:set(entry['sistema'].split('-')) for entry in plan_batches(cfg)}
    idx=0
    cif_dir=out/'03_mattergen'/'cif_validados';cif_dir.mkdir(parents=True,exist_ok=True)
    # Voy leyendo los ZIP originales, sin fabricar un CSV de candidatos a mano.
    # Cada fila conserva el lote y el fichero CIF de procedencia.
    for archive in filenames:
        batch=archive.parent.name
        if batch not in allowed:raise RuntimeError(f'Lote desconocido: {batch}')
        with zipfile.ZipFile(archive) as z:
            for name in sorted(z.namelist()):
                if not name.lower().endswith('.cif') or name.endswith('/'):continue
                idx+=1;mid=f'MG{idx:05d}';origin=f'{archive.relative_to(root)}::{name}'
                try:
                    content=z.read(name).decode('utf-8',errors='replace')
                    # Pymatgen interpreta el cristal y nos deja medir volumen, sitios
                    # y distancias para realizar filtros geométricos objetivos.
                    struct=Structure.from_str(content,fmt='cif')
                    if not struct.is_ordered:raise ValueError('Ocupaciones atómicas desordenadas')
                    if len(struct)<2 or len(struct)>256:raise ValueError('Número de sitios fuera de [2,256]')
                    chem={x.symbol for x in struct.composition.elements}
                    # Compruebo que aparecen exactamente los elementos solicitados en ese lote.
                    # Si el resultado no corresponde al sistema previsto, lo descarto y explico por qué.
                    if chem!=allowed[batch]:
                        raise ValueError(f'Sistema generado {sorted(chem)} difiere del condicionado {sorted(allowed[batch])}')
                    if struct.volume<=0 or not np.isfinite(struct.volume):raise ValueError('Volumen no positivo o no finito')
                    d=struct.distance_matrix.copy();np.fill_diagonal(d,np.inf)
                    if np.nanmin(d)<0.60:raise ValueError('Distancia interatómica <0.60 Å (filtro geométrico)')
                    formula=struct.composition.reduced_formula
                    # Entre estructuras de igual fórmula no quiero contar duplicados evidentes.
                    if any(matcher.fit(s,struct) for s in unique[formula]):
                        raise ValueError('Estructura repetida (StructureMatcher dentro de igual fórmula)')
                    unique[formula].append(struct)
                    (cif_dir/f'{mid}.cif').write_text(content,encoding='utf-8')
                    # La fila aceptada conserva la procedencia y NO se etiqueta como ensayo experimental.
                    # Lo importante es que podamos volver desde la tabla al CIF original.
                    data.append({'material_id':mid,'lote_id':batch,'formula_pretty':formula,
                        'formula_celda':struct.composition.formula.replace(' ',''),
                        'chemical_system':'-'.join(sorted(chem)), 'num_sitios':len(struct),
                        'volumen_A3':round(struct.volume,5),'novedad_formula':int(formula not in known),
                        'novedad_estructural_verificada':False,'origen_cif_zip':str(archive.relative_to(root)),
                        'nombre_cif_en_zip':name,'ruta_cif_valido':f'03_mattergen/cif_validados/{mid}.cif',
                        'origen_datos':'CIF generado MatterGen; no ensayo experimental'})
                # Si el análisis falla, anoto la fase y la traza, sin declarar éxito.
                except Exception as exc:
                    reject.append({'material_id':mid,'lote_id':batch,'origen':origin,'motivo':str(exc)[:250]})
    # Si no hay cristales reales, la evaluación se detiene.
    # No hay gráficos de resultados construidos con datos de ejemplo.
    if not idx:raise RuntimeError('No se han encontrado CIF de MatterGen')
    folder=out/'03_mattergen'
    pd.DataFrame(data,columns=['material_id','lote_id','formula_pretty','chemical_system','novedad_formula']).to_csv(folder/'candidatos_generados_validos.csv',index=False)
    pd.DataFrame(reject,columns=['material_id','lote_id','origen','motivo']).to_csv(folder/'candidatos_descartados.csv',index=False)
    if not data:raise RuntimeError('No hay CIF válidos; revisar candidatos_descartados.csv')
    df=pd.DataFrame(data)
    write_json(folder/'recuento_filtros.json',{'CIF_leidos':idx,'CIF_aceptados':len(data),
      'CIF_descartados':len(reject),'regla':'Composición exacta, orden, 2-256 sitios, volumen, distancia >=0.60 A, duplicados estructurales por fórmula'})
    # Estos recuentos por sistema darán la figura de aceptados y rechazados.
    sys_counts=[]
    for s in cfg['generacion']['sistemas']:
        expected=s['sistema'];batch_ids=[p['lote_id'] for p in plan_batches(cfg) if p['sistema']==expected]
        n=sum(sum(r['lote_id']==b for r in data) for b in batch_ids)
        nreject=sum(sum(r['lote_id']==b for r in reject) for b in batch_ids)
        sys_counts.append({'sistema':expected,'familia':s['familia'],'planificados':s['numero'],
                           'CIF_encontrados':n+nreject,'aceptados':n,'descartados':nreject})
    tab=pd.DataFrame(sys_counts);tab.to_csv(folder/'tabla_filtros_por_sistema.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.8))
    pos=np.arange(len(tab));ax.bar(pos,tab.aceptados,label='Aceptados');
    ax.bar(pos,tab.descartados,bottom=tab.aceptados,label='Descartados')
    ax.set_xticks(pos,tab.sistema);ax.set_ylabel('CIF generados');ax.set_title('Resultados reales: filtros por sistema químico')
    ax.legend();fig.tight_layout();fig.savefig(folder/'figura_filtros_por_sistema.png',dpi=160);plt.close(fig)
    return df,idx,len(reject)


# Esta es la secuencia completa del análisis. La ejecuto después
# de generar, porque necesita CIF reales y datos de referencia.
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--generados',required=True)
    p.add_argument('--salida',required=True);p.add_argument('--permitir-parcial',action='store_true')
    args=p.parse_args();cfg=load_config(args.config);gen=Path(args.generados);out=Path(args.salida)
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifiesto_analisis.json'
    tracking={'fecha_inicio':now(),'estado':'EN_CURSO','total_objetivo':100,'fases':[]}
    write_json(manifest_path,tracking)
    try:
        generation=manifest(gen,plan_batches(cfg),cfg)
        # Por defecto exijo las 100 muestras. Si hay menos, solo analizo
        # con --permitir-parcial para que la limitación quede explícita.
        if generation['cif_generados_verificados'] != 100 and not args.permitir_parcial:
            raise RuntimeError(f"Generación incompleta: {generation['cif_generados_verificados']}/100. Completar lotes o usar --permitir-parcial etiquetando la muestra.")
        (out/'configuracion_utilizada.json').write_text(Path(args.config).read_text(encoding='utf-8'),encoding='utf-8')
        tracking['generacion']={k:generation[k] for k in ('estado','muestras_programadas','cif_generados_verificados','lotes_completos')}
        tracking['fases'].append('Generación revisada')
        # Primero preparo los datos de referencia; sin ellos no hay aprendizaje supervisado.
        reference,raw,dups=get_reference(cfg,out)
        models=out/'02_modelos';models.mkdir(exist_ok=True)
        grafico_20(reference,models)
        tracking['fases'].append('Referencia Materials Project descargada/verificada')
        metrics=[];features=[]
        # Entreno dos objetivos distintos: energía sobre hull y band gap.
        # Ninguno representa directamente la conductividad del litio.
        for target in cfg['modelo_predictivo']['targets']:
            scores,oof,_,_,features=entrenar_target(reference,target,models,
                num_folds=int(cfg['modelo_predictivo']['num_folds']),
                max_rows=int(cfg['modelo_predictivo']['max_filas_entrenamiento']))
            metrics.append(scores);figura_28(oof,models,target)
        compare=pd.concat(metrics,ignore_index=True)
        compare.to_csv(models/'tabla_14_modelos_predictivos.csv',index=False)
        for target in cfg['modelo_predictivo']['targets']:
            part=compare[compare.variable_objetivo==target]
            fig,ax=plt.subplots(figsize=(8,4.5))
            ax.barh(part.modelo,part.RMSE)
            unidad='eV/átomo' if target=='energy_above_hull' else 'eV'
            ax.set_xlabel(f'RMSE ({unidad})')
            ax.set_title(f'Comparación real de error: {target}')
            fig.tight_layout();fig.savefig(models/f'figura_RMSE_{target}.png',dpi=160);plt.close(fig)
        pd.DataFrame([{'Caracteristica':'Materiales MP descargados (con solapes)','Valor':raw},
           {'Caracteristica':'Duplicados MP por ID','Valor':dups},
           {'Caracteristica':'Materiales MP limpios','Valor':len(reference)},
           {'Caracteristica':'Columnas Magpie válidas','Valor':len(features)},
           {'Caracteristica':'Semilla','Valor':cfg['semilla']},
           {'Caracteristica':'Targets supervisados','Valor':'energy_above_hull, band_gap'},
           {'Caracteristica':'Conductividad iónica calculada','Valor':'NO'},
           {'Caracteristica':'Ventana electroquímica calculada','Valor':'NO'}]
          ).to_csv(out/'01_datos'/'tabla_12_resumen.csv',index=False)
        tracking['fases'].append('Modelos entrenados y evaluados GroupKFold')
        # Tras evaluar los modelos, filtro los materiales realmente generados.
        candidates,n_raw,n_rej=screening(cfg,gen,out,reference)
        tracking['fases'].append('CIF analizados/filtrados')
        # Las predicciones se basan en composición química Magpie.
        # No equivalen a un cálculo DFT nuevo para cada estructura.
        pred=predecir_modelos(candidates.formula_pretty,models)
        candidates['energy_above_hull_predicha_eV_atom']=pred['energy_above_hull']
        candidates['band_gap_predicho_eV']=pred['band_gap']
        candidates['procedencia_predicciones']='ML composicional Magpie, no DFT individual ni medición'
        # Estas propiedades son esenciales para un electrolito, pero aquí no
        # tenemos medidas válidas. Las dejo vacías en lugar de inventarlas.
        for col in ('conductividad_ionica_S_cm','ventana_electroquimica_V',
                    'compatibilidad_interfacial_puntuacion_0_1','propiedades_mecanicas_puntuacion_0_1'):
            candidates[col]=np.nan
        selection=out/'04_seleccion';selection.mkdir(exist_ok=True)
        candidates.to_csv(selection/'tabla_13_propiedades_y_trazabilidad.csv',index=False)
        guardar_matriz_criterios(selection)
        # PCA y K-Means sirven para explorar similitudes composicionales.
        # No demuestran que los materiales transporten iones de litio.
        cluster=figuras_23_24(candidates,models,models)
        # El ranking final de esta fase es orientativo.
        # Una buena puntuación no convierte al candidato en electrolito validado.
        ranking=ranking_exploratorio(candidates,cfg,selection)
        tracking['fases'].append('Ranking exploratorio y sensibilidad calculados')
        summary={'CIF_planificados':100,'CIF_generados_y_verificados':n_raw,
          'CIF_aceptados':len(candidates),'CIF_descartados':n_rej,
          'Candidatos_en_ranking_exploratorio':len(ranking),'PCA_clustering':cluster,
          'R2_y_MAE_RMSE':'Ver tabla_14_modelos_predictivos.csv',
          'conductividad_ionica':'NO evaluada','ventana_electroquimica':'NO evaluada',
          'estabilidad_energia':'Predicha con modelo de composición; no verificada con DFT',
          'ranking':'Exploratorio, NO desempeño electroquímico demostrado',
          'distribucion':'25 por sistema (diseño exploratorio, no muestra óptima)'}
        write_json(out/'resumen_numerico.json',summary)
        lines=['# Resultados reales del experimento',f'Fecha UTC: {now()}',
               f'Muestras planificadas: 100',f'CIF encontrados: {n_raw}',
               f'CIF aceptados tras filtros: {len(candidates)}',f'CIF descartados: {n_rej}',
               f'Entradas ranking exploratorio: {len(ranking)}','',
               '## Métricas (GroupKFold por fórmula, media por pliegue)',compare.to_markdown(index=False),
               '', '## Resultados que NO se pueden afirmar',
               '- No hay conductividad iónica calculada ni medida.',
               '- Band gap electrónico no equivale a ventana electroquímica.',
               '- La energía predicha por composición no es energía DFT del cristal generado.',
               '- Estar cerca del objetivo generativo de energía no demuestra estabilidad.',
               '- Ningún candidato se considera electrolito confirmado.',
               '', '## PCA y clustering',json.dumps(cluster,ensure_ascii=False,default=str)]
        # Dejo un resumen legible para trasladar las cifras al TFM
        # y para preparar las respuestas a las preguntas del tutor.
        (out/'RESUMEN_RESULTADOS_TFM.md').write_text('\n'.join(lines),encoding='utf-8')
        tracking.update({'estado':'COMPLETADO','fecha_fin':now(),'resumen':summary})
        print('ANÁLISIS COMPLETADO',json.dumps(summary,ensure_ascii=False,default=str),flush=True)
        return 0
    except Exception as exc:
        tracking.update({'estado':'ERROR','fecha_fin':now(),'error':str(exc),'traza':traceback.format_exc()[-5000:]})
        print('ERROR ANÁLISIS:',exc,file=sys.stderr)
        return 1
    finally:write_json(manifest_path,tracking)

if __name__=='__main__':sys.exit(main())
