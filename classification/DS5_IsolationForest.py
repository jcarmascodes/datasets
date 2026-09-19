import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar datos
df = pd.read_csv('ds5_deteccion_fallas_sensores.csv')
X = df.drop(columns=['ID_Sensor', 'Estado_Sensor'])

# 2. Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Modelo Isolation Forest
iso_forest = IsolationForest(contamination=0.15, random_state=42)
pred_anomalies = iso_forest.fit_predict(X_scaled)
df['Prediccion_Anomalia'] = np.where(pred_anomalies == -1, 1, 0)

# 4. Impresión de Métricas
print("=== EVALUACIÓN DS5: ISOLATION FOREST ===")
print("Matriz de Confusión:")
print(confusion_matrix(df['Estado_Sensor'], df['Prediccion_Anomalia']))
print("\nReporte de Clasificación:")
print(classification_report(df['Estado_Sensor'], df['Prediccion_Anomalia']))

# 5. Generación de Gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica de Dispersión
sns.scatterplot(
    data=df, x='Vibracion_RMS_mm_s', y='Temperatura_C', 
    hue='Prediccion_Anomalia', style='Estado_Sensor', 
    palette={0: 'navy', 1: 'crimson'}, ax=axes[0], s=60
)
axes[0].set_title('DS5: Detección de Anomalías (Isolation Forest)', fontweight='bold')
axes[0].set_xlabel('Vibración RMS (mm/s)')
axes[0].set_ylabel('Temperatura (°C)')

# Mapa de Calor (Matriz de Confusión)
cm = confusion_matrix(df['Estado_Sensor'], df['Prediccion_Anomalia'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
            xticklabels=['Normal (0)', 'Falla (1)'], yticklabels=['Normal (0)', 'Falla (1)'])
axes[1].set_title('DS5: Matriz de Confusión (Real vs Predicho)', fontweight='bold')
axes[1].set_xlabel('Predicción (Isolation Forest)')
axes[1].set_ylabel('Estado Real (Sensor)')

plt.tight_layout()
plt.show()
