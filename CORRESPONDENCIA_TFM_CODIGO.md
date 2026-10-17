# Correspondencia TFM y código Kaggle de 100 generaciones

Documento de trazabilidad metodológica, **no resultados del experimento**.

| Apartado de memoria | Salida prevista | Advertencia |
|---|---|---|
| Tabla 11 / criterios | `04_seleccion/tabla_11_criterios_tfm.csv` | Seis criterios del estudio. Cuatro carecen de medidas independientes. |
| Tabla 12 / conjunto | `01_datos/tabla_12_resumen.csv`, `registro_descarga.json` | Descarga limitada: no muestreo representativo de MP. |
| Tabla 13 / propiedades candidatos | `04_seleccion/tabla_13_propiedades_y_trazabilidad.csv` | Campos funcionales no estimados permanecen vacíos. |
| Tabla 14 / rendimiento ML | `02_modelos/tabla_14_modelos_predictivos.csv` | MAE/RMSE/R² con validación GroupKFold por fórmula reducida. |
| Tabla 15 / pesos | `04_seleccion/tabla_15_pesos_EXPLORATORIOS.csv` | Pesos metodológicos elegidos para estudio exploratorio, no óptimos. |
| Tabla 16 / ranking | `04_seleccion/tabla_16_ranking_EXPLORATORIO.csv` | **No corresponde a electrolitos funcionales demostrados**. |
| Figura 20 / MP | `02_modelos/figura_20_distribucion_energia_MP.png` | Datos MP, no generados por IA de imagen. |
| Figuras 23/24 / PCA-clustering | `02_modelos/figura_23_*.png` y `figura_24_*.png` | Magpie por composición; si no hay muestra suficiente se omiten. |
| Figura 28 / real-estimado | `02_modelos/figura_28_real_vs_predicho_*.png` | OOF sobre datos MP; no confundir con DFT de CIF generados. |
| Figura 29 / comparación | `04_seleccion/figura_29_propiedades_exploratorias.png` | Indicadores normalizados; no conductividad. |
| Figura 30 / sensibilidad | `04_seleccion/figura_30_sensibilidad_exploratorio.png` | Cambios bajo incremento 50 % de cada peso y renormalización. |
| Generación MatterGen | `generacion/registro_lotes.csv`, `lotes/*/generated_crystals_cif.zip` | 100 = intentos; diferenciar generados, válidos, filtrados, seleccionados. |
| Conclusiones | `analisis/RESUMEN_RESULTADOS_TFM.md` | Redactar en pasado solo si está completo; citar números observados y limitaciones. |
| Anexo A.10 | Publicar archivos de código en GitHub | El enlace real lo aporta el usuario al publicar el repositorio. |

**Ajuste del alcance indispensable:** la memoria original describe la conductividad iónica y la estabilidad electroquímica como objetivos principales de un electrolito. Este proyecto **no dispone de predictores validados de esas propiedades**. Deben declararse como trabajo futuro o añadirse fuentes de datos adicionales antes de presentar el ranking como selección funcional.
