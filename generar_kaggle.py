#!/usr/bin/env python3
"""Generación verificable MatterGen (Kaggle), 100 muestras planeadas, por lotes.

No se crean CIF sintéticos ni se afirma éxito sin leer generated_crystals_cif.zip.
Cada lote usa semilla específica. Los lotes se reconocen dentro de una misma
sesión, pero el notebook v4.8 NO importa resultados de sesiones anteriores.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

# Este nombre debe coincidir con el modelo preentrenado de MatterGen.
# No significa que estemos calculando experimentalmente la energía.
MODEL='chemical_system_energy_above_hull'

# Guardo la fecha en UTC para comparar ejecuciones aunque cambie la zona horaria.
def now(): return datetime.now(timezone.utc).isoformat()
# Los manifiestos se escriben primero en un temporal.
# Así, si se corta la sesión, es menos probable dejar un JSON a medias.
def write_json(p, v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str),encoding='utf-8');tmp.replace(p)
# Antes de gastar tiempo de GPU compruebo que el plan tenga sentido.
# Aquí es donde se detectaría un reparto equivocado de las 100 muestras.
def load_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    if config['generacion']['modelo'] != MODEL: raise ValueError('Modelo MatterGen inesperado')
    systems=config['generacion']['sistemas']
    if not systems or len({s['sistema'] for s in systems})!=len(systems): raise ValueError('Sistemas no válidos o repetidos')
    for s in systems:
        if 'Li' not in s['sistema'].split('-') or int(s['numero'])<1: raise ValueError('Sistemas requieren Li y cuota >0')
    # La suma debe ser 100 exactamente: son intentos planificados, no 100 electrolitos válidos.
    if sum(int(s['numero']) for s in systems)!=100: raise ValueError('Se requieren exactamente 100 intentos (cuotas suman 100)')
    return config

# Divido el trabajo en lotes para poder recuperarlo si Kaggle se desconecta.
# Hacerlo todo de golpe sería arriesgar muchas horas de GPU.
def plan_batches(cfg):
    g=cfg['generacion']; batch_max=int(g['max_por_lote'])
    if batch_max<1: raise ValueError('max_por_lote debe ser >=1')
    result=[]
    # Cada familia recibe su cuota y cada lote una semilla distinta.
    # La semilla ayuda a repetir el ensayo, aunque CUDA no siempre es 100 % determinista.
    for sys_index,s in enumerate(g['sistemas']):
        left=int(s['numero']); batch_index=0
        while left:
            # Primer lote muy pequeño para detectar errores sin gastar una cuota grande.
            batch=(1 if sys_index==0 and batch_index==0 else min(batch_max,left))
            batch=min(batch,left)
            result.append({'sistema':s['sistema'],'familia':s['familia'],
                          'indice_sistema':sys_index,'indice_lote':batch_index+1,
                          'cantidad':batch,'semilla':int(cfg['semilla'])+sys_index*10000+batch_index,
                          'lote_id':f'{sys_index+1:02d}_{s["sistema"].replace("-","_")}_lote_{batch_index+1:02d}'})
            left-=batch;batch_index+=1
    if sum(p['cantidad'] for p in result)!=100: raise AssertionError('Recuento incorrecto')
    return result

# Aquí solo cuento CIF dentro de un ZIP que pueda leerse.
# La validez química y geométrica se comprobará después en el análisis.
def count_cif(archive):
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'ZIP CIF dañado: {bad}')
        names=[s for s in z.namelist() if s.lower().endswith('.cif') and not s.endswith('/')]
        return len(names)

# Esta es la parte delicada: lanza MatterGen o recupera un lote ya hecho.
# Prefiero parar con un error a mezclar dos ejecuciones diferentes.
def run_one(entry, cfg, root, exe, command_runner=subprocess.run):
    dest=Path(root)/'lotes'/entry['lote_id']; dest.mkdir(parents=True,exist_ok=True)
    done=dest/'lote_finalizado.json'; arch=dest/'generated_crystals_cif.zip'
    # Si existe una marca de finalizado no me fío solo de ella: verifico el ZIP real.
    if done.exists():
        record=json.loads(done.read_text(encoding='utf-8'))
        if record.get('lote_id')==entry['lote_id'] and arch.exists() and count_cif(arch)==entry['cantidad']:
            return record
        raise RuntimeError(f'Manifiesto y CIF inconsistentes: {dest}; revisar manualmente')
    # Si Kaggle cortó la ejecución, puede haber archivos a medio escribir.
    # Los aparto antes de volver a intentar el lote; no los cuento como terminados.
    if any(dest.iterdir()):
        # Solo reintento permitido cuando NO hay un ZIP resultado real en directorio incompleto.
        if arch.exists():
            found=count_cif(arch)
            if found==entry['cantidad']:
                record={**entry,'estado':'OK_RECUPERADO','cif_generados':found,'fecha_fin_utc':now(),
                        'archivo_zip_cif':str(arch.relative_to(root))}
                write_json(done,record);return record
            raise RuntimeError(f'Lote parcial {dest} ({found} CIF); sin reintento automático para no mezclar muestras')
        # La ejecución interrumpida puede dejar extxyz / logs. Se separan y archivan
        # antes de relanzar; no se mezclan resultados de distintas semillas.
        old=dest/f'intento_incompleto_{int(time.time()*1000000)}'
        old.mkdir(exist_ok=True)
        for child in list(dest.iterdir()):
            if child != old:child.rename(old/child.name)
    # Estas son condiciones que pedimos al generador, no propiedades medidas.
    # Especialmente energy_above_hull=0,05 es un objetivo, no una prueba de estabilidad.
    cond={'chemical_system':entry['sistema'],
          'energy_above_hull':float(cfg['generacion']['energia_objetivo_eV_atom'])}
    # Uso el Python del entorno propio de MatterGen porque el kernel de Kaggle
    # y la librería CUDA necesitan dependencias diferentes.
    python_mattergen=Path(exe).parent/'python'
    if not python_mattergen.is_file():
        raise RuntimeError(f'No existe el intérprete MatterGen: {python_mattergen}')
    launcher=Path(__file__).with_name('lanzar_mattergen.py')
    if not launcher.is_file():
        raise RuntimeError(f'Falta el lanzador con control de semilla: {launcher}')
    # En la orden incluyo el sistema, el lote y la semilla.
    # Así cada generación queda identificada y se puede rastrear en el informe.
    cmd=[str(python_mattergen),str(launcher),
         '--output-path',str(dest),
         '--pretrained-name',MODEL,
         '--batch-size',str(entry['cantidad']),
         '--seed',str(entry['semilla']),
         '--conditioning-json',json.dumps(cond,ensure_ascii=False),
         '--guidance',str(float(cfg['generacion']['guidance'])),
         '--record-trajectories',str(bool(cfg['generacion'].get('generar_trayectorias',False))).lower()]
    (dest/'comando_sin_secretos.txt').write_text(' '.join(cmd),encoding='utf-8')
    print(f'[{entry["lote_id"]}] Generando {entry["cantidad"]} CIF — {entry["sistema"]}',flush=True)
    started=time.monotonic()
    # Conservo la salida de consola del modelo para poder diagnosticar fallos.
    # Si el tutor pregunta qué pasó en un lote, este archivo es la evidencia.
    with (dest/'consola_mattergen.txt').open('w',encoding='utf-8',errors='replace') as log:
        process=command_runner(cmd,stdout=log,stderr=subprocess.STDOUT,check=False,env={**os.environ,'WANDB_MODE':'disabled'})
    duration=round(time.monotonic()-started,2)
    # Un comando que termina mal NO se convierte en un éxito ficticio.
    if process.returncode!=0:
        write_json(dest/'fallo.json',{**entry,'fecha':now(),'codigo_salida':process.returncode,'segundos':duration})
        raise RuntimeError(f'MatterGen falló en {entry["lote_id"]} (código {process.returncode}); consultar consola_mattergen.txt')
    # Antes de dar el lote por bueno compruebo el ZIP producido por MatterGen.
    if not arch.exists():raise RuntimeError(f'Falta generated_crystals_cif.zip en {dest}')
    n=count_cif(arch)
    if n!=entry['cantidad']:raise RuntimeError(f'{entry["lote_id"]}: se esperaban {entry["cantidad"]}, hay {n} CIF')
    # Solo registro OK cuando hay tantos CIF como se pidieron.
    # Eso todavía no significa que pasen el filtro científico posterior.
    record={**entry,'estado':'OK','cif_generados':n,'segundos':duration,
            'fecha_fin_utc':now(),'archivo_zip_cif':str(arch.relative_to(root)),
            'objetivos_de_generacion_no_medidos':cond}
    write_json(done,record)
    return record

# Este resumen junta los lotes completados y pendientes.
# Es el contador que vemos en Kaggle como 6/100, 11/100, etc.
def manifest(root,plan,cfg):
    root=Path(root)
    rows=[]
    # Al reanudar paso por los lotes en el mismo orden de siempre.
    # Los completos se verifican y se saltan; los nuevos se generan.
    for e in plan:
        p=root/'lotes'/e['lote_id']/'lote_finalizado.json'
        if p.exists():
            rec=json.loads(p.read_text(encoding='utf-8'))
            zipfile_path=root/'lotes'/e['lote_id']/'generated_crystals_cif.zip'
            if zipfile_path.exists() and count_cif(zipfile_path)==e['cantidad']:
                rows.append(rec);continue
        rows.append({**e,'estado':'PENDIENTE','cif_generados':0})
    done=sum(r.get('cif_generados',0) for r in rows if r['estado'].startswith('OK'))
    out={'fecha_utc':now(),'esquema':cfg['esquema'],'checkpoint':MODEL,
         'muestras_programadas':sum(e['cantidad'] for e in plan),'cif_generados_verificados':done,
         'lotes_totales':len(rows),'lotes_completos':sum(r['estado'].startswith('OK') for r in rows),
         'estado':'COMPLETADO' if done==100 else 'PARCIAL',
         'nota':'Objetivo energy_above_hull=0.05 es condicion generativa; no estabilidad calculada',
         'lotes':rows}
    write_json(root/'manifest_generacion.json',out)
    # El CSV de lotes es fácil de abrir en Excel y también queda vinculado al TFM.
    with (root/'registro_lotes.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['lote_id','sistema','familia','indice_lote','cantidad','semilla','estado','cif_generados','segundos']
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    return out

# Este punto de entrada conecta configuración, generación y checkpoints.
# No hay que lanzar a mano 21 comandos separados.
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--salida',required=True)
    ap.add_argument('--mattergen-exe',default=os.environ.get('MATTERGEN_EXECUTABLE','mattergen-generate'))
    ap.add_argument('--max-lotes',type=int,default=0);ap.add_argument('--checkpoint-zip',default='')
    args=ap.parse_args()
    cfg=load_config(args.config);root=Path(args.salida);root.mkdir(parents=True,exist_ok=True)
    plan=plan_batches(cfg);cmd_count=0
    try:
        for e in plan:
            check=root/'lotes'/e['lote_id']/'lote_finalizado.json'
            # Limitar cantidad de NUEVOS lotes; los ya completados no consumen el límite.
            if check.exists():
                # Verifica de nuevo el ZIP; nunca confía únicamente en la marca de finalizado.
                run_one(e,cfg,root,args.mattergen_exe)
                continue
            if args.max_lotes and cmd_count>=args.max_lotes:break
            cmd_count+=1
            run_one(e,cfg,root,args.mattergen_exe)
            result=manifest(root,plan,cfg)
            print(f'Progreso verificado: {result["cif_generados_verificados"]}/100',flush=True)
            # El lote queda guardado directamente en TFM_RESULTADOS/generacion.
            # NO subimos nada a Kaggle Datasets: la prioridad ahora es terminar.
            # Tampoco empaqueto cada lote: el ZIP único se crea al final.
        result=manifest(root,plan,cfg)
        print(json.dumps({'estado':result['estado'],'cif_generados':result['cif_generados_verificados']},ensure_ascii=False))
        return 0
    # Si falla, dejo manifestado el avance real para intentar recuperarlo después.
    except Exception as exc:
        manifest(root,plan,cfg)
        print('ERROR GENERACIÓN:',exc,file=sys.stderr)
        if args.checkpoint_zip:
            from exportar_kaggle import pack
            pack(root.parent,args.checkpoint_zip)
        return 1
if __name__=='__main__':sys.exit(main())
