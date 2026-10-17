"""Tablas/gráficos del TFM, derivados exclusivamente de CSV reales del pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd
import os
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Esta figura enseña la distribución real de energía en Materials Project.
# No sustituimos la muestra por números simulados.
def grafico_20(df, out):
    values=pd.to_numeric(df.energy_above_hull,errors='coerce').dropna()
    if values.empty: raise ValueError('No hay energy_above_hull para la Figura 20')
    # Guardo la figura como PNG para poder insertarla directamente en la memoria.
    fig,ax=plt.subplots(figsize=(7,5))
    ax.hist(values,bins=min(35,max(6,int(np.sqrt(len(values))))),edgecolor='white')
    ax.set_xlabel('Energy above hull (eV/átomo) · Materials Project')
    ax.set_ylabel('Frecuencia');ax.set_title('Figura 20. Energía en el conjunto de referencia')
    fig.tight_layout();fig.savefig(out/'figura_20_distribucion_energia_MP.png',dpi=160);plt.close(fig)


# Pongo en el eje X los valores MP y en Y las predicciones fuera de pliegue.
# La diagonal ayuda a ver cuánto se aparta el modelo de lo esperado.
def figura_28(oof, out, target):
    x=oof['real'].to_numpy();y=oof['predicho_oof'].to_numpy()
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(x,y,s=18,alpha=.65)
    lo,hi=min(x.min(),y.min()),max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'k--')
    ax.set_xlabel(f'{target}: valor de referencia MP')
    ax.set_ylabel(f'{target}: estimación fuera de muestra')
    ax.set_title('Figura 28. Real vs predicho (según objetivo)')
    fig.tight_layout();fig.savefig(out/f'figura_28_real_vs_predicho_{target}.png',dpi=160);plt.close(fig)


# PCA reduce muchos descriptores a dos ejes para visualizarlos.
# La posición no debe interpretarse como una propiedad electroquímica.
def figuras_23_24(candidatos, resultados, out):
    # PCA sobre descriptores químico-composicionales, no atribuir representatividad electroquímica.
    from modelos_tfm import descriptores_composicion
    feats=descriptores_composicion(candidatos.formula_pretty)
    if len(feats)<4: return {'estado':'omitidas','causa':'menos de 4 candidatos'}
    feats=feats.loc[:,feats.notna().any()]
    if feats.shape[1]<2: return {'estado':'omitidas','causa':'menos de 2 variables Magpie'}
    # Normalizo las variables antes del PCA: si no, una escala grande
    # podría dominar los ejes sin ser más importante científicamente.
    X=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(feats))
    coords=PCA(n_components=2).fit_transform(X)
    table=pd.DataFrame({'material_id':candidatos.material_id,'PC1':coords[:,0],'PC2':coords[:,1]})
    table.to_csv(out/'pca_candidatos.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(coords[:,0],coords[:,1],s=24,alpha=.75)
    ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_title('Figura 23. PCA de candidatos por composición Magpie')
    fig.tight_layout();fig.savefig(out/'figura_23_PCA_candidatos.png',dpi=160);plt.close(fig)
    # Pruebo varios números de grupos en K-Means y comparo silhouette.
    # Es una exploración, no una clasificación química definitiva.
    ks=range(2,min(7,len(coords)-1)+1)
    scores=[]
    for k in ks:
        lab=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(coords)
        if len(set(lab))>1:
            scores.append((k,silhouette_score(coords,lab)))
    if not scores: return {'estado':'PCA creado; clustering omitido','causa':'sin agrupación válida'}
    # Escogemos el k mejor puntuado de los probados; queda registrado
    # para explicar por qué se muestran esos grupos.
    k=max(scores,key=lambda p:p[1])[0]
    labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(coords)
    table['cluster']=labels;table.to_csv(out/'pca_y_clusters_candidatos.csv',index=False)
    pd.DataFrame(scores,columns=['k','silhouette']).to_csv(out/'silhouette_k.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(coords[:,0],coords[:,1],s=24,c=labels,cmap='viridis')
    ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_title(f'Figura 24. Agrupación K-Means de candidatos (k={k})')
    fig.tight_layout();fig.savefig(out/'figura_24_KMeans_candidatos.png',dpi=160);plt.close(fig)
    return {'estado':'PCA y clustering generados','k':k,'silhouette':dict(scores)}
