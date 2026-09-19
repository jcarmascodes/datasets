import joblib
import pandas as pd

archivo_100 = 'ds2_nuevos_100.csv'
archivo_modelo = 'model.pkl'

df_100 = pd.read_csv(archivo_100)
model = joblib.load(archivo_modelo)

X_100 = df_100.drop(columns=['ID_Orden', 'Tiempo_Ciclo_Minutos'], errors='ignore')
df_100['Prediccion_Tiempo_Ciclo_Min'] = model.predict(X_100)

df_100.to_csv('ds2_resultados_prediccion_100.csv', index=False)
print(" Predicciones guardadas en 'ds2_resultados_prediccion_100.csv'.")
print(df_100[['ID_Orden', 'Prediccion_Tiempo_Ciclo_Min']].head())
