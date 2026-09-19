import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Cargar datos
df = pd.read_csv('ds7_diagnostico_fallas_multiclase.csv')
X = df.drop(columns=['ID_Diagnostico', 'Categoria_Falla'])

# 2. Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means Multiclase (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Métricas
sil_score = silhouette_score(X_scaled, df['Cluster'])
ari_score = adjusted_rand_score(df['Categoria_Falla'], df['Cluster'])

print("=== EVALUACIÓN DS7: K-MEANS MULTICLASE (K=3) ===")
print(f"Puntuación de Silueta: {sil_score:.4f}")
print(f"Índice Rand Ajustado (ARI con Categorías Reales): {ari_score:.4f}")

# 5. Generación de Gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(
    data=df, x='Temperatura_Cojinete_C', y='Amplitud_Ruido_dB',
    hue='Cluster', style='Categoria_Falla', palette='Dark2', ax=axes[0], s=60
)
axes[0].set_title('DS7: Diagnóstico Multiclase (K-Means k=3)', fontweight='bold')
axes[0].set_xlabel('Temperatura Cojinete (°C)')
axes[0].set_ylabel('Amplitud Ruido (dB)')

ct = pd.crosstab(df['Cluster'], df['Categoria_Falla'])
sns.heatmap(ct, annot=True, fmt='d', cmap='Purples', ax=axes[1], cbar=False,
            xticklabels=['Normal (0)', 'Cojinete (1)', 'Motor (2)'], yticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'])
axes[1].set_title('DS7: Coincidencia Cluster vs Falla Real', fontweight='bold')
axes[1].set_xlabel('Categoría Falla Real')
axes[1].set_ylabel('Cluster Asignado')

plt.tight_layout()
plt.show()
