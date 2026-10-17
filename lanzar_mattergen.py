#!/usr/bin/env python3
"""Lanzador reproducible del modelo oficial MatterGen v1.0.3.

La CLI oficial 1.0.3 no expone --seed; se inicializan las RNGs de Python,
NumPy y PyTorch antes de llamar a la API de generación. Los resultados CUDA
pueden ser no deterministas pese a fijar la semilla.
"""
import argparse
import json
import random
import numpy as np
import torch
from mattergen.scripts.generate import main


# Aquí preparo una sola llamada de generación: un lote, una semilla
# y unas condiciones. Es el puente entre nuestro programa y MatterGen oficial.
def run():
    p=argparse.ArgumentParser()
    p.add_argument('--output-path',required=True)
    p.add_argument('--pretrained-name',required=True)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--conditioning-json',required=True)
    p.add_argument('--guidance',type=float,required=True)
    p.add_argument('--record-trajectories',choices=['true','false'],default='false')
    args=p.parse_args()
    # Fijo las semillas de Python, NumPy y PyTorch.
    # No prometo reproducibilidad bit a bit en cualquier GPU, pero sí un procedimiento claro.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # La condición de composición se lee desde el JSON que prepara el plan.
    # No la escribo a mano en cada ejecución para evitar despistes.
    conditioning=json.loads(args.conditioning_json)
    if not isinstance(conditioning,dict):
        raise ValueError('conditioning-json debe ser un objeto')
    print('[MATTERGEN] semilla:',args.seed,'condiciones:',conditioning,flush=True)
    # Esta llamada sí ejecuta el generador original de Microsoft.
    # Nuestro proyecto controla el proceso, pero no sustituye su algoritmo.
    main(output_path=args.output_path,pretrained_name=args.pretrained_name,
         batch_size=args.batch_size,num_batches=1,
         properties_to_condition_on=conditioning,
         diffusion_guidance_factor=args.guidance,
         record_trajectories=(args.record_trajectories=='true'))


if __name__=='__main__':
    run()
