import joblib
import pandas as pd

# 1. Cargar los 100 datos nuevos y el modelo
archivo_100 = 'ds1_nuevos_100.csv'
archivo_modelo = 'model.pkl'

df_100 = pd.read_csv(archivo_100)
model = joblib.load(archivo_modelo)

# 2. Preparar X (excluir columnas no numéricas/ID/Target si existen)
X_100 = df_100.drop(columns=['ID_Lectura', 'Consumo_Energia_KWh'], errors='ignore')

# 3. Realizar predicciones y exportar
df_100['Prediccion_Consumo_KWh'] = model.predict(X_100)

df_100.to_csv('ds1_resultados_prediccion_100.csv', index=False)
print(" Predicciones para los 100 registros guardadas en 'ds1_resultados_prediccion_100.csv'.")
print(df_100[['ID_Lectura', 'Prediccion_Consumo_KWh']].head())
