import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Cargar datos
df = pd.read_csv('ds6_control_calidad_piezas.csv')
X = df.drop(columns=['ID_Pieza', 'Pieza_Defectuosa'])

# 2. Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means (k=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Métricas de Evaluación No Supervisada
sil_score = silhouette_score(X_scaled, df['Cluster'])
ari_score = adjusted_rand_score(df['Pieza_Defectuosa'], df['Cluster'])

print("=== EVALUACIÓN DS6: K-MEANS (K=2) ===")
print(f"Puntuación de Silueta (Cohesión de Clusters): {sil_score:.4f}")
print(f"Índice Rand Ajustado (Coincidencia con Calidad Real): {ari_score:.4f}")

# 5. Generación de Gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica de Dispersión de Clusters
sns.scatterplot(
    data=df, x='Rugosidad_Superficie_um', y='Desviacion_Tolerancia_mm',
    hue='Cluster', style='Pieza_Defectuosa', palette='Set1', ax=axes[0], s=60
)
axes[0].set_title('DS6: Clustering de Calidad de Piezas (K-Means k=2)', fontweight='bold')
axes[0].set_xlabel('Rugosidad Superficial (µm)')
axes[0].set_ylabel('Desviación Tolerancia (mm)')

# Tabla cruzada visual
ct = pd.crosstab(df['Cluster'], df['Pieza_Defectuosa'])
sns.heatmap(ct, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['Aprobada (0)', 'Defectuosa (1)'], yticklabels=['Cluster 0', 'Cluster 1'])
axes[1].set_title('DS6: Coincidencia Cluster vs Calidad Real', fontweight='bold')
axes[1].set_xlabel('Calidad Real')
axes[1].set_ylabel('Cluster Asignado')

plt.tight_layout()
plt.show()
