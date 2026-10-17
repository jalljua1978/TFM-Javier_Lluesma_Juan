#!/usr/bin/env python3
"""Controlador de una ejecución nueva del TFM en Kaggle: versión 4.8.

En lugar de ir celda por celda, este controlador ejecuta la instalación,
la generación, el análisis y la exportación de resultados.

He eliminado los respaldos remotos, los tokens de Kaggle Datasets y también
los ZIP intermedios. Todos los CIF se escriben directamente en la carpeta
de resultados y el ZIP final se crea al acabar. Si falla algo, intento
conservar lo disponible en un ZIP parcial, señalando el error.

ATENCIÓN: la carpeta es temporal. Hay que descargar los resultados antes de
que Kaggle cierre o reinicie la sesión.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from exportar_kaggle import pack

# Dejamos las rutas escritas una única vez para evitar NameError entre celdas.
BASE = Path('/kaggle/working')
PROYECTO = BASE / 'TFM_KAGGLE_100'
RESULTADOS = BASE / 'TFM_RESULTADOS'
GENERACION = RESULTADOS / 'generacion'
ANALISIS = RESULTADOS / 'analisis'
ZIP_FINAL = BASE / 'Resultados_TFM_Kaggle_100.zip'
MATTERGEN = BASE / 'tfm_envs/mattergen/bin/mattergen-generate'
PY_ANALISIS = BASE / 'tfm_envs/analisis/bin/python'


def registrar(etapa: str, mensaje: str, estado: str) -> None:
    """Registro simple, sin inventar éxito cuando hay una fase incompleta."""
    ruta = RESULTADOS / 'estado_ejecucion_v48.json'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'etapa': etapa, 'estado': estado, 'mensaje': mensaje,
        'nota': 'Generar CIF no equivale a validar un electrolito sólido',
    }
    temporal = ruta.with_suffix('.tmp')
    temporal.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding='utf-8')
    temporal.replace(ruta)
    print(f'[{estado}] {etapa}: {mensaje}', flush=True)


def lanzar(*orden: object, entorno: dict | None = None) -> None:
    """Si algo falla, muestro el fallo real y no sigo como si nada."""
    print('INICIANDO:', Path(str(orden[0])).name, flush=True)
    subprocess.run([str(x) for x in orden], check=True,
                   cwd=str(PROYECTO), env=entorno)


def ejecutar_desde_cero() -> None:
    # Si ya existen cristales, no los borro. La primera celda se ocupa
    # de preparar un entorno limpio y proteger datos reales previos.
    if GENERACION.exists() and any(GENERACION.iterdir()):
        raise RuntimeError('La carpeta de generación no está vacía. No se mezcla ni sobrescribe.')
    if ZIP_FINAL.exists():
        raise RuntimeError('Hay un ZIP previo en la sesión. No se sobreescribe automáticamente.')
    PROYECTO.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    os.environ['MPLBACKEND'] = 'Agg'

    # Solo es obligatoria la API de Materials Project. No uso ninguna API de
    # datasets privados de Kaggle, ni siquiera para comprobar su existencia.
    if not os.environ.get('MP_API_KEY', '').strip():
        raise RuntimeError('Falta MP_API_KEY en Kaggle Secrets.')
    registrar('inicio', 'Nueva ejecución con 0/100 CIF; sin respaldos remotos', 'INICIADO')

    # Si la GPU está desactivada, lo sabemos enseguida, no tras instalar todo.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Activa GPU T4 en Kaggle antes de Run All.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)

    # Conservo la instalación y el generador de la rama que ya había generado
    # cristales. Lo único que suprimo aquí es el bloqueo del backup remoto.
    registrar('instalación', 'Preparando MatterGen y librerías', 'INICIADO')
    lanzar(sys.executable, PROYECTO / 'instalar_kaggle.py')
    if not MATTERGEN.is_file() or not PY_ANALISIS.is_file():
        raise RuntimeError('Faltan ejecutables después de la instalación.')
    registrar('instalación', 'Instalación verificada', 'OK')

    # Son 100 INTENTOS planeados: uno inicial para probar y después lotes de
    # hasta cinco. El contador avanza solo cuando el ZIP tiene CIF reales.
    registrar('generación', 'Comienza MatterGen desde 0/100', 'INICIADO')
    lanzar(sys.executable, PROYECTO / 'generar_kaggle.py',
           '--config', PROYECTO / 'config_tfm_kaggle.json',
           '--salida', GENERACION,
           '--mattergen-exe', MATTERGEN)

    manifest = json.loads((GENERACION / 'manifest_generacion.json').read_text(encoding='utf-8'))
    n = int(manifest.get('cif_generados_verificados', -1))
    if n != 100 or manifest.get('estado') != 'COMPLETADO':
        raise RuntimeError(f'Generación incompleta: {n}/100 CIF.')
    # Contamos de nuevo en sus ZIP originales, por separado.
    # No depende de la API de Kaggle ni de un checkpoint intermedio.
    from generar_kaggle import count_cif
    archivos = sorted((GENERACION / 'lotes').glob('*/generated_crystals_cif.zip'))
    suma = sum(count_cif(archivo) for archivo in archivos)
    if suma != 100:
        raise RuntimeError(f'Los ZIP de los lotes contienen {suma} CIF, no 100.')
    registrar('generación', '100/100 CIF reales encontrados y contados', 'OK')

    # No confundo la generación con el análisis posterior: disponer de CIF no
    # demuestra conductividad iónica ni estabilidad electroquímica medida.
    registrar('análisis', 'Comienzan modelos, filtros, PCA, gráficos y ranking', 'INICIADO')
    entorno = os.environ.copy()
    entorno['MPLBACKEND'] = 'Agg'
    estado_analisis = 'OK'
    try:
        lanzar(PY_ANALISIS, PROYECTO / 'analizar_kaggle.py',
               '--config', PROYECTO / 'config_tfm_kaggle.json',
               '--generados', GENERACION, '--salida', ANALISIS, entorno=entorno)
    except Exception as error:
        # Si falla Materials Project o un cálculo, conservo los 100 CIF.
        # El ZIP dirá que el análisis fue parcial; no crearé métricas falsas.
        estado_analisis = 'PARCIAL'
        registrar('análisis', f'Incidencia: {type(error).__name__}: {error}', 'AVISO')
    else:
        registrar('análisis', 'Análisis finalizado; revisar sus archivos', 'OK')

    # No consulto Kaggle Datasets ni la API de respaldos. Un solo ZIP final:
    # es lo que necesito para descargarlo y completar el TFM.
    registrar('final', f'CIF=100; análisis={estado_analisis}',
             'OK' if estado_analisis == 'OK' else 'PARCIAL')
    pack(RESULTADOS, ZIP_FINAL)
    print('PROCESO FINALIZADO.', flush=True)
    print('CARPETA DE RESULTADOS:', RESULTADOS, flush=True)
    print('ZIP FINAL:', ZIP_FINAL, flush=True)
    print('ATENCIÓN: descarga el ZIP antes de cerrar o perder la sesión.', flush=True)


def main() -> None:
    try:
        ejecutar_desde_cero()
    except Exception as error:
        # No dejo que el último fallo oculte el progreso anterior.
        # Si existen datos, intento dejar su ZIP local actualizado.
        print('ERROR REAL:', type(error).__name__, str(error), flush=True)
        traceback.print_exc()
        try:
            registrar('error', f'{type(error).__name__}: {error}', 'ERROR')
            # Dejo un ZIP incluso si el proceso es parcial. Al abrirlo se
            # verá el estado ERROR o PARCIAL y los lotes que existan.
            if RESULTADOS.exists():
                pack(RESULTADOS, ZIP_FINAL)
        except Exception as empaquetado:
            print('AVISO: no se pudo exportar el ZIP parcial:', empaquetado, flush=True)
        raise


if __name__ == '__main__':
    main()
