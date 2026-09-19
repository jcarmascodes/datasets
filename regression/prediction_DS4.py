import joblib
import pandas as pd

archivo_100 = 'ds4_nuevos_100.csv'
archivo_modelo = 'model.pkl'

df_100 = pd.read_csv(archivo_100)
model = joblib.load(archivo_modelo)

X_100 = df_100.drop(columns=['ID_Proceso', 'Desgaste_Herramienta_Micras'], errors='ignore')
df_100['Prediccion_Desgaste_Micras'] = model.predict(X_100)

df_100.to_csv('ds4_resultados_prediccion_100.csv', index=False)
print(" Predicciones guardadas en 'ds4_resultados_prediccion_100.csv'.")
print(df_100[['ID_Proceso', 'Prediccion_Desgaste_Micras']].head())
