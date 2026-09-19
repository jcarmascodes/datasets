import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

# 1. Cargar datos
df = pd.read_csv('ds8_anomalia_cuello_botella.csv')
X = df.drop(columns=['ID_Monitoreo', 'Anomalia_Cuello_Botella'])

# 2. Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Modelo DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=10)
df['Cluster'] = dbscan.fit_predict(X_scaled)
df['Prediccion_Ruido'] = np.where(df['Cluster'] == -1, 1, 0)

# 4. Impresión de Métricas
n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
n_noise = list(df['Cluster']).count(-1)

print("=== EVALUACIÓN DS8: DBSCAN ===")
print(f"Número de Clusters Creados: {n_clusters}")
print(f"Puntos Aislados como Ruido/Anomalía: {n_noise}")

# 5. Generación de Gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(
    data=df, x='Ocupacion_Buffer_Pct', y='Tiempo_Paro_Acumulado_Min',
    hue='Prediccion_Ruido', style='Anomalia_Cuello_Botella', palette={0: 'teal', 1: 'darkorange'}, ax=axes[0], s=60
)
axes[0].set_title('DS8: Detección de Cuellos de Botella (DBSCAN)', fontweight='bold')
axes[0].set_xlabel('Ocupación Buffer (%)')
axes[0].set_ylabel('Tiempo Paro Acumulado (min)')

cm = confusion_matrix(df['Anomalia_Cuello_Botella'], df['Prediccion_Ruido'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False,
            xticklabels=['Normal (0)', 'Ruido/Anomalía (1)'], yticklabels=['Normal (0)', 'Cuello Botella (1)'])
axes[1].set_title('DS8: Matriz de Confusión (Real vs DBSCAN Ruido)', fontweight='bold')
axes[1].set_xlabel('Predicción DBSCAN (Ruido = 1)')
axes[1].set_ylabel('Estado Real (Línea)')

plt.tight_layout()
plt.show()
