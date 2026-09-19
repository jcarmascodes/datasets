import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Cargar el Dataset y el Modelo
archivo_csv = 'ds1_consumo_energia_horno.csv'
archivo_modelo = 'model.pkl'

df = pd.read_csv(archivo_csv)
model = joblib.load(archivo_modelo)

# 2. Separar características (X) y variable objetivo (y)
col_id = 'ID_Lectura'
target_col = 'Consumo_Energia_KWh'

X = df.drop(columns=[col_id, target_col])
y_real = df[target_col]

# 3. Predicciones y cálculo de métricas
y_pred = model.predict(X)

mae = mean_absolute_error(y_real, y_pred)
mse = mean_squared_error(y_real, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_real, y_pred)

print("=== MÉTRICAS DE EVALUACIÓN - DS1 (CONSUMO ENERGÍA) ===")
print(f"MAE  (Error Absoluto Medio):  {mae:.4f} KWh")
print(f"MSE  (Error Cuadrático Medio): {mse:.4f}")
print(f"RMSE (Raíz de MSE):           {rmse:.4f} KWh")
print(f"R²   (Coeficiente R2):        {r2:.4f}")

# 4. Generación de Gráficas de Diagnóstico
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica 1: Valores Reales vs Predichos
sns.scatterplot(x=y_real, y=y_pred, alpha=0.6, color='b', ax=axes[0])
axes[0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', lw=2)
axes[0].set_title('DS1: Valores Reales vs Predicciones', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Valor Real (Consumo KWh)')
axes[0].set_ylabel('Valor Predicho (Consumo KWh)')

# Gráfica 2: Distribución de Residuales
residuales = y_real - y_pred
sns.histplot(residuales, kde=True, color='purple', ax=axes[1])
axes[1].axvline(0, color='r', linestyle='--')
axes[1].set_title('DS1: Distribución de Residuales (Errores)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Error (Real - Predicho)')

plt.tight_layout()
plt.show()
