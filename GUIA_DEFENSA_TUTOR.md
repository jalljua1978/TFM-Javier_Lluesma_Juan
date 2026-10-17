# Guía de estudio para explicar el código en la defensa del TFM

Estos son puntos para **entender y explicar**, no respuestas que deba memorizar
literalmente. Es más convincente conocer bien las decisiones y sus límites.

### 1. ¿Por qué MatterGen y no generar estructuras aleatorias?

MatterGen es un modelo generativo de materiales cristalinos preentrenado. Le
indicamos un sistema químico y una condición de energía sobre el hull. El
modelo produce candidatos en formato CIF. Aun así, cada CIF necesita filtros
y comprobaciones: una estructura generada no equivale a un electrolito válido.

### 2. ¿Por qué 100 intentos y 25 por familia?

Es un diseño exploratorio repartido de forma igualitaria para comparar cuatro
familias documentadas de conductores sólidos de litio. No es un tamaño de
muestra óptimo calculado estadísticamente. Conviene explicar también el coste
computacional disponible y la limitación de la cuota de GPU.

### 3. ¿Qué hacen `generar_kaggle.py` y `lanzar_mattergen.py`?

El primero organiza los lotes, comprueba si ya se han realizado, registra las
semillas, verifica que existan los CIF y exporta checkpoints. El segundo invoca
MatterGen con los parámetros de un único lote. Separarlos facilita saber dónde
se gestiona el experimento y dónde se ejecuta el modelo.

### 4. ¿Cómo se evita perder el avance?

Con lotes pequeños y archivos de finalización verificables. Antes de saltar un
lote, se comprueba el ZIP de CIF, no solo el texto que dice «OK». El checkpoint
se regenera tras cada lote, pero hay que descargarlo fuera de Kaggle para
protegerlo ante la pérdida de la sesión.

### 5. ¿Qué es `energy_above_hull`?

Es un indicador termodinámico respecto de la envolvente de estabilidad de fases.
La condición `0,05 eV/átomo` se entrega al modelo generativo como objetivo. La
energía que después estima el modelo de ML composicional es otra cosa: NO es
una verificación DFT individual del cristal obtenido.

### 6. ¿Qué hace Materials Project?

Proporciona un conjunto de materiales de referencia con valores calculados de
propiedades. El código descarga un subconjunto, quita IDs duplicados y prepara
los objetivos del modelo predictivo. Debo informar el número final de filas
usadas y las limitaciones de ese subconjunto.

### 7. ¿Por qué se utilizan los descriptores Magpie?

Un modelo supervisado necesita números, no una fórmula química escrita como
texto. Los descriptores Magpie calculan estadísticas composicionales a partir
de los elementos y permiten entrenar modelos sobre esos datos. No incorporan
información cristalográfica completa del CIF.

### 8. ¿Qué significan MAE, RMSE y R²?

**MAE** es el error absoluto medio; **RMSE** penaliza más los errores grandes;
**R²** indica el ajuste respecto de la variabilidad de los datos y puede ser
negativo. Las métricas se calculan con particiones de evaluación GroupKFold,
agrupando fórmulas reducidas para no mezclar composiciones equivalentes entre
entrenamiento y prueba. No deben presentarse métricas hasta que existan los
resultados reales del entrenamiento.

### 9. ¿Cómo se decide qué CIF son válidos?

Se abre cada ZIP generado y `pymatgen` interpreta el cristal. Se comprueba que
el sistema químico coincida con la condición, que la estructura tenga átomos
ordenados, volumen positivo, tamaño razonable, distancias mínimas y ausencia
de duplicados detectables por `StructureMatcher`. Son filtros iniciales;
no garantizan estabilidad termodinámica ni conductividad iónica.

### 10. ¿Para qué se usan PCA y K-Means?

Para visualizar semejanzas entre candidatos basadas en sus descriptores
composicionales. PCA reduce dimensiones; K-Means agrupa puntos y el coeficiente
silhouette ayuda a elegir entre varios números de grupos. Los clústeres no son
«familias de electrolitos comprobadas».

### 11. ¿Cómo funciona el ranking?

Normaliza tres indicadores disponibles a una escala 0–1 y combina sus valores
con pesos exploratorios de 0,65, 0,25 y 0,10. Una energía estimada menor favorece
la puntuación; un mayor band gap o una fórmula no presente en el subconjunto
MP también puntúan. **Esto no prueba que el material sea buen electrolito:**
faltan propiedades funcionales fundamentales.

### 12. ¿Qué aporta la sensibilidad de pesos?

Repite la clasificación dando un 50 % más de peso a cada criterio por separado
y renormalizando. Si los primeros puestos cambian mucho, el ranking depende
fuertemente de las decisiones de ponderación y se debe reconocer.

### 13. ¿Qué significa SHA256 y por qué se corrigió el ZIP?

SHA256 es una huella digital de cada archivo. Sirve para detectar cambios
posteriores. La versión anterior introducía a veces dos archivos con el mismo
nombre de auditoría; la nueva exportación excluye la auditoría vieja, crea
una nueva y comprueba nombres únicos, CRC y hashes.

### 14. ¿Cuáles son las principales limitaciones científicas?

No calculamos directamente difusión de Li+, conductividad iónica, ventana
electroquímica, interfaces o propiedades mecánicas. El ranking debe describirse
como **preselección exploratoria**. Para conclusiones funcionales harían falta
métodos adicionales, cálculos y/o validaciones experimentales.

### 15. ¿Qué enseñaría para demostrar trazabilidad?

Un ID de material en `tabla_16`, su registro en `tabla_13`, el CIF original
en el ZIP de un lote, su entrada en el manifiesto de generación, y el CSV/PNG
real que demuestra el resultado. Los valores aún no disponibles deben
aparecer como ausentes y no como ceros inventados.

---

**Nota de preparación:** cuando finalice Kaggle, sustituir todas las explicaciones
genéricas por números observados: CIF generados, descartes por motivo, conjunto MP,
mejor modelo y sus métricas, número final de candidatos y sensibilidad real.
