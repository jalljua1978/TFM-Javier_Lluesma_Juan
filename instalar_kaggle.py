#!/usr/bin/env python3
"""Instalador MatterGen Kaggle v4.4: CUDA y fuentes oficiales completas.

No modifica el kernel de Kaggle. Aísla los índices para impedir el conflicto
requests/pymatgen/wandb al resolver contra el índice de PyTorch.
La ejecución real en Kaggle debe verificarse externamente.
"""
from __future__ import annotations
import os, shutil, subprocess, sys
from pathlib import Path

# Fijo las fuentes de los paquetes porque mezclar índices fue uno de
# los problemas encontrados al probar la instalación anterior.
TORCH_INDEX='https://download.pytorch.org/whl/cu118'
PYG_WHEELS='https://data.pyg.org/whl/torch-2.2.0+cu118.html'
PYPI_INDEX='https://pypi.org/simple'
ROOT=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
ENVS=ROOT/'tfm_envs'


# Kaggle hereda algunas variables del entorno. Las limpio para evitar que
# una instalación busque dependencias en un repositorio que no corresponde.
def env_uv():
    """Evita que índices de CUDA anteriores contaminen la resolución actual."""
    env=os.environ.copy()
    for k in ('UV_INDEX','UV_INDEX_URL','UV_EXTRA_INDEX_URL','UV_DEFAULT_INDEX',
              'UV_FIND_LINKS','UV_INDEX_STRATEGY','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL',
              'PIP_FIND_LINKS'):
        env.pop(k,None)
    # Kaggle exporta un backend inline del kernel. El venv separado no tiene
    # matplotlib_inline; un proceso no interactivo debe usar Agg.
    env['MPLBACKEND']='Agg'
    env['UV_LINK_MODE']='copy'
    env['UV_INDEX_STRATEGY']='first-index'
    return env


# Esta función pequeña ejecuta las instalaciones y detiene todo si hay un fallo.
# No tendría sentido generar materiales con medio entorno instalado.
def run(args,**kwargs):
    args=list(map(str,args))
    print('>', ' '.join(args),flush=True)
    subprocess.run(args,check=True,env=env_uv(),**kwargs)


# PyTorch es la base de los cálculos en GPU. Uso la versión CUDA
# que resulta compatible con las dependencias de MatterGen de este proyecto.
def comando_torch(uv,py_mg):
    return [uv,'pip','install','--python',py_mg,
            'torch==2.2.1+cu118','torchvision==0.17.1+cu118',
            'torchaudio==2.2.1+cu118','--index-url',TORCH_INDEX]


# PyTorch Geometric necesita extensiones compiladas compatibles.
# Si se descargan ruedas equivocadas, la GPU puede fallar al empezar.
def comando_pyg(uv,py_mg):
    # Las extensiones nativas deben proceder únicamente de ruedas PyG para
    # torch 2.2 y cu118: evita compilarlas desde fuentes en la T4.
    return [uv,'pip','install','--python',py_mg,'--no-index','--no-deps',
            '--only-binary',':all:','--find-links',PYG_WHEELS,
            'pyg_lib','torch_scatter','torch_sparse','torch_cluster']


# Separo PyPI del índice CUDA para evitar conflictos de versiones.
# Aquí aprendí que instalar todo con el mismo índice puede dar errores extraños.
def comando_mattergen(uv,py_mg):
    # CLAVE: no pasar --extra-index-url ni --index URL de PyTorch.
    # PyTorch y PyG ya están instalados; MatterGen usa solo PyPI para
    # resolver requests>=2.32, pymatgen, wandb y demás dependencias.
    return [uv,'pip','install','--python',py_mg,
            'mattergen==1.0.3','numpy<2','setuptools<81',
            '--index-url',PYPI_INDEX]


# Primero compruebo que Kaggle realmente tenga una GPU disponible.
# El portátil local no tiene NVIDIA, por eso este cálculo va en la nube.
def comprobar_gpu_kernel():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Kaggle no tiene PyTorch en el kernel. Selecciona GPU T4 x2.') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('GPU no disponible: Settings > Accelerator > GPU T4 x2. Reinicia sesión.')
    print('[GPU Kaggle]',torch.cuda.get_device_name(0),'(',round(torch.cuda.get_device_properties(0).total_memory/2**30,1),'GiB )',flush=True)


# No basta con que pip diga «instalado»: reviso CUDA y versiones
# desde el propio intérprete que utilizará MatterGen.
def comprobar_entorno(py_mg):
    script=(
        'import importlib.metadata as m,torch,numpy,torch_scatter,torch_sparse,torch_cluster,mattergen; '
        'assert torch.cuda.is_available(), "CUDA no disponible para MatterGen"; '
        'assert torch.__version__.startswith("2.2.1+cu118"),torch.__version__; '
        'assert int(numpy.__version__.split(".")[0])<2,numpy.__version__; '
        'assert m.version("mattergen")=="1.0.3",m.version("mattergen"); '
        'print("[OK] MatterGen GPU comprobado:",torch.__version__,torch.cuda.get_device_name(0))'
    )
    run([py_mg,'-c',script])


# Este arreglo es importante: la rueda publicada omitía archivos de
# configuración; por eso usamos el repositorio oficial en un tag concreto.
def instalar_fuente_oficial(uv, py_mg):
    """La rueda PyPI 1.0.3 omite sampling_conf y gemnet-dT.json.

    Usamos el repositorio oficial fijado al tag v1.0.3. Instalación editable
    SIN dependencias para preservar el torch cu118 comprobado y lo ya instalado.
    """
    repo=ROOT/'tfm_mattergen_fuente_oficial_v1_0_3'
    if not (repo/'.git').is_dir():
        if repo.exists():
            raise RuntimeError(f'Hay una carpeta incompleta en {repo}; revisarla antes de continuar')
        env=env_uv()
        env['GIT_LFS_SKIP_SMUDGE']='1'  # HuggingFace descargará el checkpoint al requerirlo.
        cmd=['git','clone','--depth','1','--single-branch','--branch','v1.0.3',
             'https://github.com/microsoft/mattergen.git',str(repo)]
        print('>', ' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True,env=env)
    # Sin estas configuraciones, MatterGen no sabe cómo iniciar el muestreo.
    if not (repo/'sampling_conf/default.yaml').is_file():
        raise RuntimeError('El repositorio oficial no incluye sampling_conf/default.yaml')
    if not (repo/'mattergen/common/gemnet/gemnet-dT.json').is_file():
        raise RuntimeError('El repositorio oficial no incluye gemnet-dT.json')
    run([uv,'pip','install','--python',py_mg,'--no-deps','-e',str(repo),
         '--index-url',PYPI_INDEX])
    print('[OK] Código oficial MatterGen v1.0.3 editable: recursos completos',flush=True)


# Compruebo expresamente los archivos que faltaban en la instalación v4.3.
# Es mejor localizar el problema aquí que después de esperar la GPU.
def comprobar_archivos_configuracion(py_mg):
    script=(
        'from pathlib import Path; '
        'from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH; '
        'import mattergen; '
        'from hydra import initialize_config_dir, compose; '
        'p=Path(DEFAULT_SAMPLING_CONFIG_PATH); '
        'q=Path(mattergen.__file__).resolve().parent/"common/gemnet/gemnet-dT.json"; '
        'assert (p/"default.yaml").is_file(),f"Falta {p}/default.yaml"; '
        'assert q.is_file(),f"Falta {q}"; '
        'print("[OK] sampling_conf:",p); '
        'print("[OK] GemNet:",q)'
    )
    run([py_mg,'-c',script])


# Instalo en dos entornos independientes, uno para generar y otro para analizar.
# Así evitamos que las bibliotecas científicas se pisen entre sí.
def install():
    comprobar_gpu_kernel()
    uv=shutil.which('uv')
    if not uv:
        run([sys.executable,'-m','pip','install','uv'])
        uv=shutil.which('uv') or str(Path.home()/'.local/bin/uv')
    if not Path(uv).is_file() and not shutil.which(uv):
        raise RuntimeError('No se ha encontrado uv después de instalarlo')
    run([uv,'python','install','3.10'])
    mg=ENVS/'mattergen';analisis=ENVS/'analisis'
    for path in (mg,analisis):
        if not (path/'bin/python').is_file():
            run([uv,'venv',str(path),'--python','3.10','--seed'])
    py_mg=str(mg/'bin/python');py_a=str(analisis/'bin/python')
    # El orden de instalación está pensado para fijar primero CUDA
    # y después añadir MatterGen sin reemplazar ese núcleo.
    print('[1/4] PyTorch CUDA 11.8: índice exclusivo de PyTorch',flush=True)
    run(comando_torch(uv,py_mg))
    print('[2/4] Extensiones PyTorch Geometric: ruedas compiladas cu118',flush=True)
    run(comando_pyg(uv,py_mg))
    print('[3/4] MatterGen y librerías generales: SOLO PyPI',flush=True)
    run(comando_mattergen(uv,py_mg))
    instalar_fuente_oficial(uv, py_mg)
    comprobar_entorno(py_mg)
    comprobar_archivos_configuracion(py_mg)
    exe=mg/'bin/mattergen-generate'
    if not exe.is_file():
        raise RuntimeError('MatterGen no instaló el ejecutable mattergen-generate')
    # La segunda parte necesita pandas, matminer, pymatgen y scikit-learn
    # para trabajar con datos y validar los CIF.
    print('[4/4] Dependencias del análisis',flush=True)
    run([uv,'pip','install','--python',py_a,'-r',Path(__file__).parent/'requirements_analisis.txt',
         '--index-url',PYPI_INDEX])
    run([py_a,'-c','import numpy,pandas,sklearn,pymatgen,matminer,mp_api,matplotlib;print("[OK] Análisis disponible")'])
    print('[OK] INSTALACIÓN LISTA. El proceso automático sigue con los lotes nuevos hasta 100 CIF.',flush=True)

if __name__=='__main__':
    install()
