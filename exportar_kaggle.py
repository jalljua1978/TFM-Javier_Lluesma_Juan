#!/usr/bin/env python3
"""Exporta los resultados del TFM con una auditoría SHA256 clara y sin duplicados.

Este archivo no genera cristales ni inventa resultados: solo empaqueta lo que
existe en el directorio de resultados en el momento de llamar a ``pack``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path


# Aquí dejo agrupados los nombres que nunca deberían acabar en un ZIP público.
# La clave de Materials Project se usa desde los secretos de Kaggle, no se copia.
NOMBRES_SENSIBLES = {'.env', 'MP_API_KEY', 'credentials.json'}
ARCHIVO_AUDITORIA = 'AUDITORIA_SHA256.json'


def sha256_bytes(data: bytes) -> str:
    """Calcula la huella de un fichero; basta un cambio para que sea diferente."""
    return hashlib.sha256(data).hexdigest()


def incluir_en_zip(fichero: Path, raiz: Path, destino: Path) -> bool:
    """Decide qué entra en el paquete y qué conviene dejar fuera."""
    if not fichero.is_file() or fichero.is_symlink():
        return False

    # Muy importante: no metemos el propio ZIP dentro de sí mismo.
    # Tampoco la auditoría antigua. Esta última era la causa del warning v4.4.
    if fichero.resolve() == destino.resolve():
        return False
    if fichero.name == ARCHIVO_AUDITORIA or fichero.name in NOMBRES_SENSIBLES:
        return False

    # No hace falta enviar cachés ni intentos incompletos. Se mantienen en
    # Kaggle para diagnosticar fallos, pero complicarían innecesariamente el ZIP.
    ruta_relativa = fichero.relative_to(raiz)
    partes = set(ruta_relativa.parts)
    if '__pycache__' in partes or '.ipynb_checkpoints' in partes:
        return False
    if any(parte.startswith('intento_incompleto_') for parte in partes):
        return False
    if fichero.name.endswith('.zip.tmp'):
        return False

    # MatterGen puede generar trayectorias muy grandes. Para las tablas del TFM
    # usamos las estructuras finales, que ya vienen en los ZIP de cada lote.
    if fichero.name in {'generated_crystals.extxyz', 'generated_trajectories.zip'}:
        return False
    if 'cif_validados' in partes and fichero.suffix.lower() == '.cif':
        return False
    return True


def pack(raiz, archivo_zip):
    """Crea una exportación nueva; se puede llamar varias veces sin duplicar."""
    root = Path(raiz)
    target = Path(archivo_zip)
    root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Primero inventariamos los resultados reales. La auditoría anterior queda
    # fuera siempre, así que nunca se calcula el hash de una auditoría caducada.
    items = []
    for child in sorted(root.rglob('*')):
        if not incluir_en_zip(child, root, target):
            continue
        contenido = child.read_bytes()
        items.append({
            'ruta': child.relative_to(root).as_posix(),
            'sha256': sha256_bytes(contenido),
            'bytes': len(contenido),
        })

    # La fecha es la de esta exportación. No se presenta como fecha de síntesis
    # ni como prueba de que todos los cálculos hayan terminado correctamente.
    audit = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'n_archivos': len(items),
        'archivos': items,
        'nota': ('El ZIP contiene los resultados disponibles, incluso si la '
                 'ejecución es parcial. La auditoría no certifica validez científica.'),
    }
    audit_path = root / ARCHIVO_AUDITORIA
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')

    # Uso un archivo temporal: si Kaggle corta la exportación a mitad,
    # no sustituimos el último ZIP completo por otro incompleto.
    temporal = target.with_name(target.name + '.tmp')
    try:
        with zipfile.ZipFile(temporal, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for item in items:
                z.write(root / item['ruta'], arcname=item['ruta'])
            z.write(audit_path, arcname=ARCHIVO_AUDITORIA)

        # Una vez terminado, comprobamos tanto el ZIP como el inventario.
        # Así sabemos que no hemos vuelto a añadir dos veces el mismo nombre.
        with zipfile.ZipFile(temporal, 'r') as z:
            nombres = z.namelist()
            if len(nombres) != len(set(nombres)):
                raise RuntimeError('El ZIP contiene nombres repetidos')
            dañado = z.testzip()
            if dañado is not None:
                raise RuntimeError(f'Archivo comprimido dañado: {dañado}')
            for item in items:
                contenido = z.read(item['ruta'])
                if len(contenido) != item['bytes'] or sha256_bytes(contenido) != item['sha256']:
                    raise RuntimeError(f'Fallo en auditoría de: {item["ruta"]}')

        # Solo ahora hacemos el cambio definitivo. Un nuevo checkpoint pasa
        # a reemplazar al anterior, pero únicamente después de verificarlo.
        os.replace(temporal, target)
    finally:
        if temporal.exists():
            temporal.unlink()

    print(f'ZIP EXPORTADO: {target} '
          f'({target.stat().st_size / 1024 / 1024:.1f} MB, {len(items)} archivos)')
    return audit


def main():
    # Dejo esta interfaz para poder exportar también desde una terminal:
    # python exportar_kaggle.py --raiz resultados --zip resultados.zip
    parser = argparse.ArgumentParser()
    parser.add_argument('--raiz', required=True)
    parser.add_argument('--zip', required=True)
    args = parser.parse_args()
    pack(args.raiz, args.zip)


if __name__ == '__main__':
    main()
