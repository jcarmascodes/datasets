import joblib
import pandas as pd

archivo_100 = 'ds3_nuevos_100.csv'
archivo_modelo = 'model.pkl'

df_100 = pd.read_csv(archivo_100)
model = joblib.load(archivo_modelo)

X_100 = df_100.drop(columns=['ID_Registro', 'Demanda_Estimada_Unidades'], errors='ignore')
df_100['Prediccion_Demanda_Unidades'] = model.predict(X_100)

df_100.to_csv('ds3_resultados_prediccion_100.csv', index=False)
print(" Predicciones guardadas en 'ds3_resultados_prediccion_100.csv'.")
print(df_100[['ID_Registro', 'Prediccion_Demanda_Unidades']].head())
