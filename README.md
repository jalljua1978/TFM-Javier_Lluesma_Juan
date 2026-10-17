# Cómo poner en marcha el proyecto MatterGen (versión 4.8)

## ¿Qué hace este proyecto?

En este TFM he preparado un pequeño flujo de trabajo para generar y analizar estructuras cristalinas con **MatterGen**, un modelo de inteligencia artificial generativa aplicado a materiales. La idea es explorar posibles candidatos relacionados con electrolitos sólidos para baterías de litio.

El programa está pensado para intentar generar **100 estructuras** repartidas entre cuatro sistemas químicos: **Li–P–S, Li–La–Zr–O, Li–In–Cl y Li–Al–Ti–P–O** (25 por sistema). Después, si la generación termina correctamente, pasa a la parte de análisis, modelos predictivos, gráficos y clasificación de candidatos.

Aquí conviene aclarar algo desde el principio: **generar un archivo CIF no significa haber descubierto un electrolito sólido válido**. Un CIF describe una estructura cristalina; la utilidad real del material necesita estudiar otras propiedades y, en última instancia, una validación que va más allá de este programa. Los resultados del proyecto son **computacionales y exploratorios**.

## ¿Por qué se ejecuta en Kaggle?

Elegí Kaggle porque el equipo con el que trabajo no dispone de una tarjeta gráfica adecuada para ejecutar este tipo de generación de manera razonable. Kaggle permite utilizar GPU desde el navegador y ejecutar cuadernos de Python sin tener que preparar todo el entorno en mi ordenador.

La ventaja es que casi todo está automatizado: **se importa un notebook, se configura Kaggle y se pulsa `Run All`**. No hace falta abrir diez programas diferentes ni ejecutar los archivos Python uno por uno.

## ¿Cuál es el archivo que tengo que abrir?

El punto de entrada es el notebook:

`TFM_MatterGen_Kaggle_RUN_ALL_SIN_RESPALDOS_v4_8.ipynb`

Este es el archivo que hay que **importar en Kaggle**. En el repositorio también hay varios archivos `.py`, pero no son diez programas independientes que el usuario tenga que poner en marcha manualmente. Cada uno se encarga de una tarea concreta y el notebook los prepara y los llama en el orden adecuado.

Dentro de esos archivos, **`ejecutar_todo_kaggle.py`** es el coordinador principal: organiza la instalación, la generación, el análisis y la exportación. Dicho de forma sencilla, el notebook es la puerta de entrada y `ejecutar_todo_kaggle.py` es el que va dando paso a cada fase.

## Antes de empezar: lo que hace falta

- Una cuenta de **Kaggle** y acceso a la sección **Code / Notebooks**.
- **GPU habilitada** en la configuración del notebook (en este proyecto se ha usado **GPU T4 x2**).
- **Internet activado**, porque deben descargarse bibliotecas y recursos.
- Una clave de **Materials Project**, guardada en Kaggle como secreto con el nombre exacto **`MP_API_KEY`**.
- Algo de paciencia: instalar el entorno y generar estructuras con IA puede tardar bastante.

**Importante:** la clave de Materials Project **no se escribe dentro del notebook**, ni en los archivos `.py`, ni se sube a GitHub. Se guarda en **Add-ons → Secrets** de Kaggle y se autoriza su uso desde el cuaderno.

## Ejecución paso a paso

### 1. Importar el notebook

Entro en [Kaggle Code](https://www.kaggle.com/code), creo un notebook nuevo y cargo el archivo `.ipynb` indicado arriba. Conviene comprobar que en el encabezado aparece **v4.8** para no ejecutar por equivocación una versión anterior.

### 2. Preparar Kaggle

En la configuración del notebook selecciono una **GPU T4 x2**, activo **Internet** y compruebo que el secreto `MP_API_KEY` está creado y autorizado. Para esta versión **no hacen falta `KAGGLE_API_TOKEN` ni `KAGGLE_USERNAME`**, porque eliminé el sistema de respaldos remotos.

### 3. Iniciar la sesión

Enciendo la **Draft Session** y espero a que aparezca el **punto verde**. Eso indica que Jupyter está listo para ejecutar Python.

### 4. Pulsar `Run All`

Cuando la sesión ya está activa, pulso **`Run All` una sola vez**. A partir de aquí el notebook ejecuta las celdas en orden: escribe los scripts de apoyo en Kaggle, instala lo necesario, comprueba la GPU y pone en marcha el controlador.

En esta versión **uso `Run All` en la sesión interactiva**, no `Save Version → Save & Run All`. No hay que ir pulsando cada celda ni volver a ejecutar el cuaderno mientras está trabajando.

### 5. Seguir el progreso

Durante la generación aparecen mensajes de consola parecidos a estos:

```text
[INICIADO] generación: Comienza MatterGen desde 0/100
[01_Li_P_S_lote_01] Generando 1 CIF - Li-P-S
Progreso verificado: 1/100
[01_Li_P_S_lote_02] Generando 5 CIF - Li-P-S
Progreso verificado: 6/100
```

Los números son solo **un ejemplo del formato del registro**, no resultados que haya que dar por hechos en otra ejecución. El contador avanza cuando termina un lote y se comprueba que sus CIF existen. Por eso puede pasar un rato sin que aparezca ningún mensaje nuevo: no significa automáticamente que el programa esté bloqueado.

### 6. Recoger los resultados

Si todo termina correctamente, Kaggle deja los resultados en:

```text
/kaggle/working/TFM_RESULTADOS
```

Ahí se organiza el material generado. En particular:

- `TFM_RESULTADOS/generacion/`: lotes, estructuras CIF y registros de generación.
- `TFM_RESULTADOS/analisis/`: tablas, métricas, gráficos y clasificaciones **si el análisis se ha completado**.

El proyecto también intenta crear el archivo:

```text
/kaggle/working/Resultados_TFM_Kaggle_100.zip
```

Este ZIP reúne los resultados disponibles para descargarlos de manera más cómoda. Desde el **panel de archivos / Output de Kaggle** puedo buscar esos archivos y descargarlos a mi ordenador. Si el panel no aparece, hay que abrir el explorador de archivos del entorno de Kaggle. **No basta con guardar o descargar el propio notebook**: los CIF y los CSV son archivos distintos.

## ¿Qué hace cada archivo del proyecto?

| Archivo | Para qué sirve (explicado sin complicarlo demasiado) |
|---|---|
| `TFM_MatterGen_Kaggle_RUN_ALL_SIN_RESPALDOS_v4_8.ipynb` | **Es el que abro y ejecuto en Kaggle.** Contiene las celdas que preparan el proyecto. |
| `ejecutar_todo_kaggle.py` | **Es el coordinador.** Va poniendo en marcha las distintas fases y comprueba si han terminado. |
| `instalar_kaggle.py` | Instala MatterGen y las librerías que hacen falta. |
| `generar_kaggle.py` | Organiza los intentos de generación, los lotes y el recuento de CIF. |
| `lanzar_mattergen.py` | Realiza la llamada al generador MatterGen con los parámetros de cada lote. |
| `config_tfm_kaggle.json` | Guarda los sistemas químicos y otros parámetros del experimento. |
| `preparar_materiales.py` | Agrupa definiciones que utiliza el tratamiento de materiales y datos de referencia. |
| `analizar_kaggle.py` | Coordina la parte de análisis de los materiales generados. |
| `modelos_tfm.py` | Prepara y evalúa los modelos predictivos. |
| `graficos_tfm.py` | Crea figuras a partir de los datos calculados. |
| `ranking_tfm.py` | Organiza la evaluación y clasificación exploratoria de candidatos. |
| `exportar_kaggle.py` | Prepara el ZIP de resultados y la información de integridad. |
| `requirements_analisis.txt` | Enumera las librerías necesarias para el análisis. |

Me interesaba separar las tareas de esta manera porque, si hay un problema, es bastante más sencillo saber en qué parte ocurre. Además, al tutor le resulta más fácil revisar el código que si estuviera absolutamente todo mezclado en un único archivo enorme.

## Algunas cosas que conviene tener en cuenta

**No hay respaldo remoto automático en la versión v4.8.** Se eliminó expresamente para simplificar la ejecución. Los resultados se guardan **dentro de la sesión de Kaggle**, así que si el entorno se reinicia o se pierde, también pueden perderse archivos. Por eso es importante **descargar los resultados antes de cerrar la sesión**, y, si es posible, conservar copias parciales durante la ejecución.

Tampoco hay que dar por supuesto que se obtendrán exactamente 100 CIF válidos. El objetivo son 100 estructuras generadas, pero la ejecución puede fallar o terminar parcialmente. El programa distingue entre la generación de estructuras y su análisis posterior. Incluso aunque se generen 100 CIF, puede haber materiales que no superen los filtros o propiedades que no se hayan podido calcular.

Si aparece un error, lo mejor es **no volver a pulsar `Run All` a ciegas**. Primero miro el mensaje de la consola, compruebo lo que hay guardado en `/kaggle/working/TFM_RESULTADOS` y, si existe un ZIP parcial, intento descargarlo. Así evito borrar o sobrescribir resultados aprovechables.

## Resumiéndolo en pocas palabras

Si alguien me preguntara cómo usar el programa, lo explicaría así:

> Abres el notebook v4.8 en Kaggle, activas la GPU e Internet, autorizas la clave de Materials Project, inicias la sesión y pulsas `Run All`. El cuaderno prepara los scripts, instala MatterGen, intenta generar las estructuras, realiza los análisis que pueda completar y deja los resultados en una carpeta y en un ZIP. Después hay que descargarlos, porque la sesión de Kaggle no es un almacenamiento permanente.

Esa es la idea principal del proyecto: **que otra persona pueda ejecutar el mismo procedimiento, consultar los scripts comentados y comprobar de dónde salen las tablas y las gráficas del TFM**, sin tener que reconstruir todos los pasos manualmente.
