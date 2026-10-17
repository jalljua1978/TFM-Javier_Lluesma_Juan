# Comprobaciones de la versión v4.8

## Qué se ha comprobado localmente

- El notebook es un fichero Jupyter válido con 18 celdas (14 de código).
- Todas las celdas de Python que escriben scripts y el controlador compilan sintácticamente.
- La configuración planifica exactamente **100 intentos de generación** en 21 lotes y 4 familias químicas.
- Mediante un generador de pruebas (sin GPU y SIN datos científicos) se ha probado el reconocimiento de 100 CIF ficticios repartidos en los 21 lotes y su manifiesto.
- El exportador se ha ejecutado dos veces y se ha comprobado que no duplica nombres dentro del ZIP, incluida `AUDITORIA_SHA256.json`.
- El notebook no llama a la API de Kaggle Datasets: no crea ni comprueba datasets privados y no requiere `KAGGLE_API_TOKEN` ni `KAGGLE_USERNAME`.

## Lo que NO se ha comprobado

- No se ha ejecutado MatterGen realmente desde este entorno con una GPU de Kaggle.
- No se ha verificado la descarga del modelo ni la instalación completa dentro de la sesión actual de Kaggle.
- No se han obtenido aquí resultados científicos nuevos, modelos validados ni resultados de Materials Project.

**Los datos sintéticos de prueba no están incluidos en el proyecto ni deben utilizarse en el TFM.**

## Al ejecutar en Kaggle

Usar **Draft Session + Run All**. Los datos se crearán en `/kaggle/working/TFM_RESULTADOS`. Se exportará `/kaggle/working/Resultados_TFM_Kaggle_100.zip` cuando termine, o cuando el controlador pueda empaquetar una ejecución parcial tras un error. No hay copias fuera de la sesión.
