import datetime # to handle time

def dateparse(time_in_secs):
    """Convierte timestamps a objetos datetime"""
    return datetime.datetime.fromtimestamp(float(time_in_secs))

from auto_implementations.implementations import np, pd, plt, train_test_split, RandomForestRegressor

print("Cargando datos...")
# Importar el dataset y codificar la fecha
df = pd.read_csv('coin_Bitcoin.csv', parse_dates=[0], date_parser=dateparse)
print("Datos cargados. Mostrando las últimas 3 filas:")
df.tail(3)

print("Limpiando datos...")

# Primero, arreglamos los datos para las barras/velas donde no hay operaciones.
# Volumen/operaciones son eventos únicos, así que rellenamos "NaN" con ceros para los campos relevantes...
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
df['Weighted_Price'].fillna(value=0, inplace=True)

# En segundo lugar, necesitamos arreglar los datos OHLC (open, high, low, close) que son una serie temporal continua,
# así que vamos a rellenar hacia adelante esos valores.
df['Open'].fillna(method='ffill', inplace=True)
df['High'].fillna(method='ffill', inplace=True)
df['Low'].fillna(method='ffill', inplace=True)
df['Close'].fillna(method='ffill', inplace=True)
print("Datos limpios. Mostrando las últimas 3 filas:")
df.tail(3)

# Indicar el tiempo de 'inicio' y 'fin'. [Del principio al final. (END puede cambiarse según la fecha actual)]
start = datetime.datetime(2009, 1, 1, 0, 0, 0, 0)
end = datetime.datetime(2020, 10, 17, 20, 0, 0, 0)

print("Agrupando datos por día...")
# Encontrar filas entre el tiempo de inicio y fin y encontrar la primera fila (lunes por la mañana a las 00:00)
# Agrupar por día
daily_df = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)].groupby([pd.Grouper(key='Timestamp', freq='D')]).first().reset_index()
daily_df = daily_df.set_index('Timestamp')
print("Datos agrupados. Mostrando las últimas 3 filas:")
daily_df.tail(3)

print("Limpiando datos diarios...")
# Rellenar NaN nuevamente
daily_df['Volume_(BTC)'].fillna(value=0, inplace=True)
daily_df['Volume_(Currency)'].fillna(value=0, inplace=True)
daily_df['Weighted_Price'].fillna(value=0, inplace=True)

daily_df['Open'].fillna(method='ffill', inplace=True)
daily_df['High'].fillna(method='ffill', inplace=True)
daily_df['Low'].fillna(method='ffill', inplace=True)
daily_df['Close'].fillna(method='ffill', inplace=True)

print("Forma: ", daily_df.shape)
print("Columnas= ", daily_df.columns)
print("¿Hay algún valor 'NaN'?: ", daily_df.isnull().values.any())
print("¿Hay algún valor duplicado?: ", daily_df.index.duplicated().any() if hasattr(daily_df.index, 'duplicated') else "No se puede verificar")
print("Datos diarios limpios. Mostrando las últimas 3 filas:")
daily_df.tail(3)

print("Añadiendo datos históricos...")
# Añadir datos históricos
historical_df = daily_df
for i in range(1, 8):  # para 7 días
    historical_df["Open_b_" + str(i)] = daily_df['Open'].shift(i)
    historical_df["High_b_" + str(i)] = daily_df['High'].shift(i)
    historical_df["Low_b_" + str(i)] = daily_df['Low'].shift(i)
    historical_df["Close_b_" + str(i)] = daily_df['Close'].shift(i)
    historical_df["Volume_(BTC)_b_" + str(i)] = daily_df['Volume_(BTC)'].shift(i)
    historical_df["Volume_(Currency)_b_" + str(i)] = daily_df['Volume_(Currency)'].shift(i)

historical_df = historical_df.dropna()  # eliminar las primeras filas. No tienen información previa
print("Forma de Datos Históricos: ", historical_df.shape)
print("Datos históricos añadidos. Mostrando las últimas 3 filas:")
historical_df.tail(3)

print("Añadiendo etiqueta NEXT_CLOSE...")
# Añadir Etiqueta
historical_df["NEXT_CLOSE"] = historical_df['Close'].shift(-1)  # Añadir datos siguientes como etiqueta para datos actuales
historical_df = historical_df.dropna()  # eliminar la última fila. No tiene información siguiente
print("Después de añadir la etiqueta NEXT_CLOSE, nueva forma:", historical_df.shape)
print("Etiqueta añadida. Mostrando las últimas 3 filas:")
historical_df.tail(3)

print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
# Dividir Datos y establecer los datos de prueba/entrenamiento y sus etiquetas
prediction_days = 140
df_train = historical_df[:len(historical_df) - prediction_days]
df_test = historical_df[len(historical_df) - prediction_days:]

print("PORCENTAJE datos de prueba/datos totales = %", (prediction_days / len(historical_df)) * 100)
print("Forma de datos de entrenamiento:", df_train.shape)
print("Forma de datos de prueba:", df_test.shape)

training_set = df_train.values
X_train = training_set[0:len(training_set), 0:49]
y_train = training_set[0:len(training_set), 49].reshape(-1, 1)

test_set = df_test.values
X_test = test_set[0:len(test_set), 0:49]
y_test = test_set[0:len(test_set), 49].reshape(-1, 1)

print("Primeras 3 filas de datos de entrenamiento:")
df_train.head(3)

print("Entrenando modelo Random Forest...")
# ENTRENAR MODELO
rf = RandomForestRegressor(n_estimators=1000, random_state=5)
rf.fit(X_train, np.ravel(y_train))
print("Modelo entrenado. Realizando predicciones...")
predictions = rf.predict(X_test)

# Calcular los errores absolutos
errors = np.sqrt(np.mean(np.square([p - a[0] for p, a in zip(predictions, y_test)])))
# Imprimir el error cuadrático medio (rmse)
print('RMSE:', errors)
print('Puntuación R^2 - Coeficiente de Determinación', rf.score(X_test, y_test))

print("Visualizando resultados...")
# Fusionar los datos predichos y reales
df_Result = pd.DataFrame(y_test, index=df_test.index, columns=["NEXT_CLOSE"])
df_Result['Predicted'] = predictions
df_Result = df_Result.sort_values('Timestamp')
print("Últimas 3 filas de resultados:")
df_Result.tail(3)

# Visualizar los resultados
plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()
plt.plot(df_Result['NEXT_CLOSE'], color='red', label='Precio Real de BTC')
plt.plot(df_Result['Predicted'], color='blue', label='Precio Predicho de BTC')
plt.title('Predicción de Precio de BTC', fontsize=40)
df_test = df_Result.reset_index()
x = df_test.index
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Tiempo', fontsize=40)
plt.ylabel('Precio de BTC(USD) [Cerrado]', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()

print("Análisis completo de predicción de precios de Bitcoin finalizado.")