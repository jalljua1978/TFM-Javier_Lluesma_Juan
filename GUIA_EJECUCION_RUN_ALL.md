# Kaggle: ejecución sencilla de MatterGen v4.8

1. En Kaggle Code, **importar el notebook nuevo**, sin reemplazar el anterior.
2. Activar GPU T4 (una o dos), **Internet On** y autorizar **solo `MP_API_KEY`** en *Add-ons → Secrets*.
3. Encender **Draft Session** y esperar el indicador verde.
4. Pulsar **Run All** una vez. El notebook escribe sus scripts, instala los entornos y ejecuta los lotes hasta completar 100 intentos; después analiza y exporta.
5. **No** pulsar `Save & Run All` para esta modalidad, ni lanzar otra ejecución mientras está trabajando.
6. Al terminar, abrir en el panel de archivos de Kaggle `/kaggle/working/TFM_RESULTADOS` y descargar **`/kaggle/working/Resultados_TFM_Kaggle_100.zip`**. Si no aparece el ZIP final, revisar si hubo error y solicitar ayuda antes de cerrar la sesión.

**No hay respaldo remoto.** El almacenamiento de la sesión es temporal: si Kaggle se reinicia durante la generación, el avance de la sesión podría perderse. Si la sesión dura hasta el final, la carpeta y el ZIP final estarán disponibles para descargar. Este documento no garantiza la finalización ni la persistencia de los archivos si la sesión termina.
