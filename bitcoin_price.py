import datetime # to handle time

def dateparse(date_string):
    """Convierte fechas en diferentes formatos a objetos datetime"""
    try:
        # Intentar parsear como timestamp numérico
        return datetime.datetime.fromtimestamp(float(date_string))
    except ValueError:
        try:
            # Intentar parsear como fecha formateada
            return datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                # Intentar parsear solo fecha
                return datetime.datetime.strptime(date_string, '%Y-%m-%d')
            except ValueError:
                print(f"[ERROR] No se pudo parsear la fecha: {date_string}")
                return None

def map_columns_to_expected(df):
    """
    Mapea las columnas del CSV real a las esperadas por el script
    """
    # Mapeo de columnas del CSV real a las esperadas
    column_mapping = {
        'Date': 'Timestamp',
        'High': 'High',
        'Low': 'Low', 
        'Open': 'Open',
        'Close': 'Close',
        'Volume': 'Volume_(BTC)',
        'Marketcap': 'Volume_(Currency)'
    }
    
    new_data = {}
    
    # Mapear columnas existentes
    for old_col, new_col in column_mapping.items():
        if old_col in df._data:
            new_data[new_col] = df._data[old_col]
    
    # Crear Weighted_Price como promedio de High y Low
    if 'High' in df._data and 'Low' in df._data:
        weighted_prices = []
        for i in range(len(df._data['High'].data)):
            high = df._data['High'].data[i]
            low = df._data['Low'].data[i]
            if high is not None and low is not None:
                weighted_prices.append((high + low) / 2)
            else:
                weighted_prices.append(None)
        
        from auto_implementations.implementations import MySeries
        new_data['Weighted_Price'] = MySeries(weighted_prices, df.index, 'Weighted_Price')
    
    from auto_implementations.implementations import MyDataFrame
    return MyDataFrame(new_data, df.index, list(new_data.keys()))

from auto_implementations.implementations import np, pd, plt, train_test_split, RandomForestRegressor

print("Cargando datos...")
# Importar el dataset y codificar la fecha
df_raw = pd.read_csv('coin_Bitcoin.csv', parse_dates=[3], date_parser=dateparse)  # Cambiar índice a 3 para 'Date'
df = map_columns_to_expected(df_raw)  # Mapear columnas
print("Datos cargados. Mostrando las últimas 3 filas:")
print(df.tail(3))

print("Limpiando datos...")

# Primero, arreglamos los datos para las barras/velas donde no hay operaciones.
# Volumen/operaciones son eventos únicos, así que rellenamos "NaN" con ceros para los campos relevantes...
if 'Volume_(BTC)' in df._data:
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
if 'Volume_(Currency)' in df._data:
    df['Volume_(Currency)'].fillna(value=0, inplace=True)
if 'Weighted_Price' in df._data:
    df['Weighted_Price'].fillna(value=0, inplace=True)

# En segundo lugar, necesitamos arreglar los datos OHLC (open, high, low, close) que son una serie temporal continua,
# así que vamos a rellenar hacia adelante esos valores.
if 'Open' in df._data:
    df['Open'].fillna(method='ffill', inplace=True)
if 'High' in df._data:
    df['High'].fillna(method='ffill', inplace=True)
if 'Low' in df._data:
    df['Low'].fillna(method='ffill', inplace=True)
if 'Close' in df._data:
    df['Close'].fillna(method='ffill', inplace=True)

print("Datos limpios. Mostrando las últimas 3 filas:")
print(df.tail(3))

# Indicar el tiempo de 'inicio' y 'fin'. [Del principio al final. (END puede cambiarse según la fecha actual)]
start = datetime.datetime(2013, 4, 29, 0, 0, 0, 0)  # Ajustar a fecha real del dataset
end = datetime.datetime(2021, 5, 30, 20, 0, 0, 0)   # Ajustar a fecha real del dataset

print("Agrupando datos por día...")
# Encontrar filas entre el tiempo de inicio y fin y encontrar la primera fila
# Agrupar por día
try:
    filtered_df = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]
    daily_df = filtered_df.groupby([pd.Grouper(key='Timestamp', freq='D')]).first().reset_index()
    daily_df = daily_df.set_index('Timestamp')
    print("Datos agrupados. Mostrando las últimas 3 filas:")
    print(daily_df.tail(3))
except Exception as e:
    print(f"Error en agrupación: {e}")
    # Fallback: usar datos sin agrupar
    daily_df = df.set_index('Timestamp')

print("Limpiando datos diarios...")
# Rellenar NaN nuevamente
if 'Volume_(BTC)' in daily_df._data:
    daily_df['Volume_(BTC)'].fillna(value=0, inplace=True)
if 'Volume_(Currency)' in daily_df._data:
    daily_df['Volume_(Currency)'].fillna(value=0, inplace=True)
if 'Weighted_Price' in daily_df._data:
    daily_df['Weighted_Price'].fillna(value=0, inplace=True)

if 'Open' in daily_df._data:
    daily_df['Open'].fillna(method='ffill', inplace=True)
if 'High' in daily_df._data:
    daily_df['High'].fillna(method='ffill', inplace=True)
if 'Low' in daily_df._data:
    daily_df['Low'].fillna(method='ffill', inplace=True)
if 'Close' in daily_df._data:
    daily_df['Close'].fillna(method='ffill', inplace=True)

print("Forma: ", daily_df.shape)
print("Columnas= ", daily_df.columns)

# Verificación de valores nulos mejorada
try:
    null_check = daily_df.isnull().values
    has_nulls = any(any(row) for row in null_check) if null_check else False
    print("¿Hay algún valor 'NaN'?: ", has_nulls)
except Exception as e:
    print(f"Error verificando valores nulos: {e}")
    print("¿Hay algún valor 'NaN'?: No se puede verificar")

# Verificación de duplicados mejorada
try:
    if hasattr(daily_df.index, 'duplicated'):
        has_duplicates = daily_df.index.duplicated().any()
        print("¿Hay algún valor duplicado?: ", has_duplicates)
    else:
        print("¿Hay algún valor duplicado?: No se puede verificar")
except Exception as e:
    print("¿Hay algún valor duplicado?: No se puede verificar")

print("Datos diarios limpios. Mostrando las últimas 3 filas:")
print(daily_df.tail(3))

print("Añadiendo datos históricos...")
# Añadir datos históricos
historical_df = daily_df.copy() if hasattr(daily_df, 'copy') else daily_df

# Verificar que las columnas existen antes de crear características históricas
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
available_columns = [col for col in required_columns if col in daily_df._data]

for i in range(1, 8):  # para 7 días
    for col in available_columns:
        if col in daily_df._data:
            historical_df[f"{col}_b_{i}"] = daily_df[col].shift(i)

historical_df = historical_df.dropna()  # eliminar las primeras filas. No tienen información previa
print("Forma de Datos Históricos: ", historical_df.shape)
print("Datos históricos añadidos. Mostrando las últimas 3 filas:")
print(historical_df.tail(3))

print("Añadiendo etiqueta NEXT_CLOSE...")
# Añadir Etiqueta
if 'Close' in historical_df._data:
    historical_df["NEXT_CLOSE"] = historical_df['Close'].shift(-1)  # Añadir datos siguientes como etiqueta para datos actuales
    historical_df = historical_df.dropna()  # eliminar la última fila. No tiene información siguiente
    print("Después de añadir la etiqueta NEXT_CLOSE, nueva forma:", historical_df.shape)
    print("Etiqueta añadida. Mostrando las últimas 3 filas:")
    print(historical_df.tail(3))
else:
    print("Error: No se encontró la columna 'Close' para crear la etiqueta")

print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
# Dividir Datos y establecer los datos de prueba/entrenamiento y sus etiquetas
prediction_days = min(140, len(historical_df) // 4)  # Ajustar si hay pocos datos
df_train = historical_df[:len(historical_df) - prediction_days]
df_test = historical_df[len(historical_df) - prediction_days:]

print("PORCENTAJE datos de prueba/datos totales = %", (prediction_days / len(historical_df)) * 100)
print("Forma de datos de entrenamiento:", df_train.shape)
print("Forma de datos de prueba:", df_test.shape)

# Verificar que tenemos suficientes datos
if df_train.shape[0] == 0 or df_test.shape[0] == 0:
    print("Error: No hay suficientes datos para entrenar el modelo")
    exit()

training_set = df_train.values
test_set = df_test.values

# Ajustar índices según el número real de columnas
num_features = len(historical_df.columns) - 1  # Restar 1 por la columna NEXT_CLOSE
# Convertir datos a formato manejable
print("Convirtiendo datos de entrenamiento...")
training_data = []
for i in range(len(df_train.index)):
    row = []
    for col in df_train.columns:
        if col in df_train._data and i < len(df_train._data[col].data):
            row.append(df_train._data[col].data[i])
        else:
            row.append(None)
    training_data.append(row)

print("Convirtiendo datos de prueba...")
test_data = []
for i in range(len(df_test.index)):
    row = []
    for col in df_test.columns:
        if col in df_test._data and i < len(df_test._data[col].data):
            row.append(df_test._data[col].data[i])
        else:
            row.append(None)
    test_data.append(row)

# Extraer características y etiquetas
num_features = len(historical_df.columns) - 1  # Restar 1 por la columna NEXT_CLOSE
print(f"Número de características: {num_features}")

# Crear matrices de entrenamiento
X_train = []
y_train = []
for row in training_data:
    if len(row) > num_features and row[num_features] is not None:
        X_train.append(row[:num_features])
        y_train.append([row[num_features]])  # Mantener como lista para compatibilidad

# Crear matrices de prueba
X_test = []
y_test = []
for row in test_data:
    if len(row) > num_features and row[num_features] is not None:
        X_test.append(row[:num_features])
        y_test.append([row[num_features]])

print(f"Datos de entrenamiento: {len(X_train)} filas x {len(X_train[0]) if X_train else 0} características")
print(f"Etiquetas de entrenamiento: {len(y_train)} filas")
print(f"Datos de prueba: {len(X_test)} filas x {len(X_test[0]) if X_test else 0} características")
print(f"Etiquetas de prueba: {len(y_test)} filas")

# Verificar que tenemos datos válidos
if not X_train or not y_train or not X_test or not y_test:
    print("Error: No hay suficientes datos válidos para entrenar el modelo")
    exit()

print("Primeras 3 filas de datos de entrenamiento:")
print(df_train.head(3))

print("Entrenando modelo Random Forest...")
# ENTRENAR MODELO
rf = RandomForestRegressor(n_estimators=100, random_state=5)  # Reducir estimadores para velocidad
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

# Verificar si se puede ordenar por Timestamp
try:
    df_Result = df_Result.sort_values('Timestamp')
except:
    print("No se puede ordenar por Timestamp, usando orden original")

print("Últimas 3 filas de resultados:")
print(df_Result.tail(3))

# Visualizar los resultados
plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()
plt.plot(df_Result['NEXT_CLOSE'], color='red', label='Precio Real de BTC')
plt.plot(df_Result['Predicted'], color='blue', label='Precio Predicho de BTC')
plt.title('Predicción de Precio de BTC', fontsize=40)

# Configurar ejes
try:
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
except:
    print("No se pudo configurar el tamaño de fuente de los ejes")

plt.xlabel('Tiempo', fontsize=40)
plt.ylabel('Precio de BTC(USD) [Cerrado]', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()

print("Análisis completo de predicción de precios de Bitcoin finalizado.")