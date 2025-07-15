import math
import random
import datetime
import csv
from collections import defaultdict

# ====================== NUMPY IMPLEMENTACIÓN ======================
class MyNumpy:
    @staticmethod
    def ravel(arr):
        """
        Aplana un array multidimensional a 1D
        
        Parámetros:
        - arr: array a aplanar
        
        Retorna:
        - array aplanado
        """
        if not isinstance(arr, list):
            return arr
        
        result = []
        for item in arr:
            if isinstance(item, list):
                result.extend(MyNumpy.ravel(item))
            else:
                result.append(item)
        return result
    
    @staticmethod
    def mean(arr):
        """
        Calcula la media de un array
        
        Parámetros:
        - arr: array de números
        
        Retorna:
        - media aritmética
        """
        return sum(arr) / len(arr)
    
    @staticmethod
    def square(arr):
        """
        Eleva al cuadrado cada elemento del array
        
        Parámetros:
        - arr: array de números
        
        Retorna:
        - array con elementos al cuadrado
        """
        if isinstance(arr, list):
            return [x**2 for x in arr]
        return arr**2
    
    @staticmethod
    def sqrt(value):
        """
        Calcula la raíz cuadrada
        
        Parámetros:
        - value: número para calcular raíz cuadrada
        
        Retorna:
        - raíz cuadrada del valor
        """
        return math.sqrt(value)
    
    @staticmethod
    def reshape(arr, shape):
        """
        Cambia la forma de un array
        
        Parámetros:
        - arr: array a cambiar de forma
        - shape: tupla con la nueva forma
        
        Retorna:
        - array con nueva forma
        """
        if shape[0] == -1:
            # Si es -1, calculamos automáticamente la primera dimensión
            if len(shape) > 1:
                dim2 = shape[1]
                dim1 = len(arr) // dim2
                return [arr[i*dim2:(i+1)*dim2] for i in range(dim1)]
            return arr
        
        # Para el caso específico reshape(-1,1) que se usa en el código
        if shape == (-1, 1):
            if isinstance(arr, list):
                return [[x] for x in arr]
            return [[arr]]
        
        return arr

# ====================== PANDAS IMPLEMENTACIÓN ======================
class MySeries:
    def __init__(self, data=None, index=None, name=None):
        """
        Inicializa una serie de datos
        
        Parámetros:
        - data: datos de la serie
        - index: índice de la serie
        - name: nombre de la serie
        """
        self.data = data if data is not None else []
        self.index = index if index is not None else list(range(len(self.data)))
        self.name = name
        self.values = self.data  # Para acceso directo a los valores
    
    def __getitem__(self, key):
        """Permite acceso por índice o slice"""
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self.data)
            step = key.step if key.step is not None else 1
            
            indices = list(range(start, stop, step))
            new_data = [self.data[i] for i in indices if i < len(self.data)]
            new_index = [self.index[i] for i in indices if i < len(self.index)]
            
            return MySeries(new_data, new_index, self.name)
        elif isinstance(key, int):
            return self.data[key]
        
        return None
    
    def __ge__(self, other):
        """
        Implementa el operador >= para comparaciones
        
        Parámetros:
        - other: valor a comparar
        
        Retorna:
        - MySeries con resultados booleanos
        """
        result = []
        for value in self.data:
            if value is None:
                result.append(False)
            else:
                result.append(value >= other)
        return MySeries(result, self.index, self.name)
    
    def __le__(self, other):
        """
        Implementa el operador <= para comparaciones
        
        Parámetros:
        - other: valor a comparar
        
        Retorna:
        - MySeries con resultados booleanos
        """
        result = []
        for value in self.data:
            if value is None:
                result.append(False)
            else:
                result.append(value <= other)
        return MySeries(result, self.index, self.name)
    
    def __and__(self, other):
        """
        Implementa el operador & para operaciones lógicas
        
        Parámetros:
        - other: otra serie para operación AND
        
        Retorna:
        - MySeries con resultados booleanos
        """
        result = []
        for i in range(len(self.data)):
            if i < len(other.data):
                result.append(self.data[i] and other.data[i])
            else:
                result.append(False)
        return MySeries(result, self.index, self.name)
    
    def fillna(self, value=None, method=None, inplace=False):
        """
        Rellena valores nulos en la serie
        
        Parámetros:
        - value: valor para rellenar
        - method: método de relleno ('ffill' para forward fill)
        - inplace: si se modifica la serie original
        
        Retorna:
        - serie con valores rellenados
        """
        new_data = self.data.copy()
        
        if method == 'ffill':
            # Forward fill: propaga el último valor válido
            last_valid = None
            for i in range(len(new_data)):
                if new_data[i] is None or (isinstance(new_data[i], float) and math.isnan(new_data[i])):
                    if last_valid is not None:
                        new_data[i] = last_valid
                else:
                    last_valid = new_data[i]
        elif value is not None:
            # Rellenar con un valor específico
            for i in range(len(new_data)):
                if new_data[i] is None or (isinstance(new_data[i], float) and math.isnan(new_data[i])):
                    new_data[i] = value
        
        if inplace:
            self.data = new_data
            return self
        else:
            return MySeries(new_data, self.index.copy(), self.name)
    
    def shift(self, periods=1):
        """
        Desplaza los datos por el número de periodos especificado
        
        Parámetros:
        - periods: número de periodos a desplazar (positivo hacia adelante, negativo hacia atrás)
        
        Retorna:
        - nueva serie con datos desplazados
        """
        if periods == 0:
            return MySeries(self.data.copy(), self.index.copy(), self.name)
        
        new_data = [None] * len(self.data)
        
        if periods > 0:
            # Desplazamiento hacia adelante (valores pasados)
            for i in range(len(self.data) - periods):
                new_data[i + periods] = self.data[i]
        else:
            # Desplazamiento hacia atrás (valores futuros)
            periods = abs(periods)
            for i in range(periods, len(self.data)):
                new_data[i - periods] = self.data[i]
        
        return MySeries(new_data, self.index.copy(), self.name)

class MyDataFrame:
    def __init__(self, data=None, index=None, columns=None):
        """
        Inicializa un DataFrame
        
        Parámetros:
        - data: datos del DataFrame (diccionario, lista de listas, etc.)
        - index: índice del DataFrame
        - columns: nombres de columnas
        """
        self.columns = columns if columns is not None else []
        self._data = {}  # Almacena las series por columna
        
        if isinstance(data, dict):
            # Si los datos son un diccionario
            for col, values in data.items():
                if isinstance(values, MySeries):
                    self._data[col] = values
                else:
                    self._data[col] = MySeries(values, index, col)
            
            if not self.columns:
                self.columns = list(data.keys())
        elif isinstance(data, list) and data and isinstance(data[0], list):
            # Si los datos son una lista de listas (matriz)
            if not columns:
                columns = [f'col_{i}' for i in range(len(data[0]))]
            
            for col_idx, col_name in enumerate(columns):
                col_data = [row[col_idx] for row in data if col_idx < len(row)]
                self._data[col_name] = MySeries(col_data, index, col_name)
            
            self.columns = columns
        elif data is not None:
            # Caso especial para y_test en el código
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and len(data[0]) == 1:
                # Es una lista de listas de un solo elemento (como y_test.reshape(-1,1))
                flat_data = [item[0] for item in data]
                if columns and len(columns) == 1:
                    self._data[columns[0]] = MySeries(flat_data, index, columns[0])
                    self.columns = columns
            else:
                # Otros casos
                if columns and len(columns) == 1:
                    self._data[columns[0]] = MySeries(data, index, columns[0])
                    self.columns = columns
        
        # Establecer el índice
        self.index = index if index is not None else []
        if not self.index and self._data:
            # Usar el índice de la primera serie si no se proporciona uno
            first_col = next(iter(self._data.values()))
            self.index = first_col.index
    
    def __getitem__(self, key):
        """
        Acceso a columnas o filas del DataFrame
        
        Parámetros:
        - key: nombre de columna, lista de nombres, slice o condición booleana
        
        Retorna:
        - Serie o DataFrame según el caso
        """
        if isinstance(key, str):
            # Acceso a una columna
            return self._data.get(key, MySeries())
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            # Acceso a múltiples columnas
            new_data = {col: self._data[col] for col in key if col in self._data}
            return MyDataFrame(new_data, self.index, key)
        elif isinstance(key, slice):
            # Acceso a filas por slice
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self.index)
            step = key.step if key.step is not None else 1
            
            new_index = self.index[start:stop:step]
            new_data = {}
            
            for col in self.columns:
                new_data[col] = self._data[col][key]
            
            return MyDataFrame(new_data, new_index, self.columns)
        elif isinstance(key, MySeries):
            # Filtrado por condición booleana
            valid_indices = []
            for i, value in enumerate(key.data):
                if value:
                    valid_indices.append(i)
            
            new_index = [self.index[i] for i in valid_indices]
            new_data = {}
            
            for col in self.columns:
                new_data[col] = MySeries([self._data[col].data[i] for i in valid_indices], new_index, col)
            
            return MyDataFrame(new_data, new_index, self.columns)
        
        return None
    
    def set_index(self, column):
        """
        Establece una columna como índice
        
        Parámetros:
        - column: nombre de la columna a usar como índice
        
        Retorna:
        - DataFrame con el nuevo índice
        """
        if column not in self._data:
            return self
        
        new_index = self._data[column].data
        new_columns = [col for col in self.columns if col != column]
        new_data = {col: self._data[col] for col in new_columns}
        
        return MyDataFrame(new_data, new_index, new_columns)
    
    def reset_index(self):
        """
        Resetea el índice a números enteros consecutivos
        
        Retorna:
        - DataFrame con índice reseteado
        """
        new_index = list(range(len(self.index)))
        new_data = {}
        
        for col in self.columns:
            new_data[col] = MySeries(self._data[col].data, new_index, col)
        
        # Añadir el índice anterior como una columna si no es numérico consecutivo
        if not all(isinstance(idx, int) and idx == i for i, idx in enumerate(self.index)):
            new_data['Timestamp'] = MySeries(self.index, new_index, 'Timestamp')
            new_columns = ['Timestamp'] + self.columns
        else:
            new_columns = self.columns
        
        return MyDataFrame(new_data, new_index, new_columns)
    
    def sort_values(self, by):
        """
        Ordena el DataFrame por una columna
        
        Parámetros:
        - by: nombre de la columna para ordenar
        
        Retorna:
        - DataFrame ordenado
        """
        if by not in self._data:
            return self
        
        # Crear pares (valor, índice) para ordenar
        pairs = [(self._data[by].data[i], i) for i in range(len(self.index))]
        pairs.sort(key=lambda x: x[0] if x[0] is not None else float('-inf'))
        
        # Reordenar datos
        new_index = [self.index[i] for _, i in pairs]
        new_data = {}
        
        for col in self.columns:
            new_data[col] = MySeries([self._data[col].data[i] for _, i in pairs], new_index, col)
        
        return MyDataFrame(new_data, new_index, self.columns)
    
    def tail(self, n=5):
        """
        Devuelve las últimas n filas
        
        Parámetros:
        - n: número de filas a devolver
        
        Retorna:
        - DataFrame con las últimas n filas
        """
        if len(self.index) <= n:
            return self
        
        start_idx = len(self.index) - n
        new_index = self.index[start_idx:]
        new_data = {}
        
        for col in self.columns:
            new_data[col] = MySeries(self._data[col].data[start_idx:], new_index, col)
        
        return MyDataFrame(new_data, new_index, self.columns)
    
    def head(self, n=5):
        """
        Devuelve las primeras n filas
        
        Parámetros:
        - n: número de filas a devolver
        
        Retorna:
        - DataFrame con las primeras n filas
        """
        if len(self.index) <= n:
            return self
        
        new_index = self.index[:n]
        new_data = {}
        
        for col in self.columns:
            new_data[col] = MySeries(self._data[col].data[:n], new_index, col)
        
        return MyDataFrame(new_data, new_index, self.columns)
    
    def dropna(self):
        """
        Elimina filas con valores nulos
        
        Retorna:
        - DataFrame sin filas con valores nulos
        """
        valid_indices = []
        
        # Encontrar índices de filas válidas
        for i in range(len(self.index)):
            is_valid = True
            for col in self.columns:
                val = self._data[col].data[i]
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(i)
        
        # Crear nuevo DataFrame
        new_index = [self.index[i] for i in valid_indices]
        new_data = {}
        
        for col in self.columns:
            new_data[col] = MySeries([self._data[col].data[i] for i in valid_indices], new_index, col)
        
        return MyDataFrame(new_data, new_index, self.columns)
    
    def isnull(self):
        """
        Comprueba valores nulos
        
        Retorna:
        - DataFrame de booleanos indicando valores nulos
        """
        new_data = {}
        
        for col in self.columns:
            null_values = []
            for val in self._data[col].data:
                is_null = val is None or (isinstance(val, float) and math.isnan(val))
                null_values.append(is_null)
            
            new_data[col] = MySeries(null_values, self.index, col)
        
        return MyDataFrame(new_data, self.index, self.columns)
    
    @property
    def values(self):
        """
        Devuelve los valores como una matriz
        
        Retorna:
        - Lista de listas con los valores del DataFrame
        """
        result = []
        
        for i in range(len(self.index)):
            row = []
            for col in self.columns:
                row.append(self._data[col].data[i])
            result.append(row)
        
        return result
    
    @property
    def shape(self):
        """
        Devuelve la forma del DataFrame
        
        Retorna:
        - Tupla (filas, columnas)
        """
        return (len(self.index), len(self.columns))
    
    @staticmethod
    def read_csv(filepath, parse_dates=None, date_parser=None):
        """
        Lee un archivo CSV y devuelve un DataFrame
        
        Parámetros:
        - filepath: ruta del archivo
        - parse_dates: lista de índices de columnas a parsear como fechas
        - date_parser: función para parsear fechas
        
        Retorna:
        - DataFrame con los datos del CSV
        """
        data = {}
        index = []
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Inicializar columnas
                for header in headers:
                    data[header] = []
                
                # Leer datos
                for row in reader:
                    for i, value in enumerate(row):
                        if i < len(headers):  # Asegurarse de no exceder los headers
                            header = headers[i]
                            
                            # Convertir fechas si es necesario
                            if parse_dates and i in parse_dates and date_parser:
                                try:
                                    parsed_value = date_parser(value)
                                    data[header].append(parsed_value)
                                except:
                                    data[header].append(None)
                            else:
                                # Intentar convertir a número
                                try:
                                    numeric_value = float(value)
                                    # Convertir a entero si es posible
                                    if numeric_value.is_integer():
                                        numeric_value = int(numeric_value)
                                    data[header].append(numeric_value)
                                except:
                                    data[header].append(value)
            
            # Crear DataFrame
            return MyDataFrame(data, columns=headers)
        
        except Exception as e:
            print(f"Error al leer el archivo CSV: {e}")
            return MyDataFrame()

class MyGrouper:
    def __init__(self, key, freq):
        """
        Inicializa un agrupador por tiempo
        
        Parámetros:
        - key: columna a agrupar
        - freq: frecuencia ('D' para día)
        """
        self.key = key
        self.freq = freq

class MyGroupBy:
    def __init__(self, df, by):
        """
        Inicializa un objeto GroupBy
        
        Parámetros:
        - df: DataFrame a agrupar
        - by: columna o lista de columnas para agrupar
        """
        self.df = df
        self.by = by
        self.groups = self._create_groups()
    
    def _create_groups(self):
        """
        Crea los grupos según la columna especificada
        
        Retorna:
        - Diccionario de grupos
        """
        groups = defaultdict(list)
        
        if isinstance(self.by, list) and len(self.by) == 1 and isinstance(self.by[0], MyGrouper):
            # Caso especial para pd.Grouper
            grouper = self.by[0]
            key_col = grouper.key
            
            if grouper.freq == 'D':
                # Agrupar por día
                for i, timestamp in enumerate(self.df.index):
                    # Extraer solo la fecha (sin hora)
                    if hasattr(timestamp, 'date'):
                        date_key = timestamp.date()
                        groups[date_key].append(i)
                    elif isinstance(timestamp, datetime.datetime):
                        date_key = timestamp.date()
                        groups[date_key].append(i)
        else:
            # Agrupar por columna normal
            by_col = self.by[0] if isinstance(self.by, list) else self.by
            
            for i, value in enumerate(self.df[by_col].data):
                groups[value].append(i)
        
        return groups
    
    def first(self):
        """
        Devuelve el primer valor de cada grupo
        
        Retorna:
        - DataFrame con los primeros valores de cada grupo
        """
        new_data = {}
        new_index = []
        
        for group_key, indices in self.groups.items():
            if indices:
                first_idx = indices[0]
                new_index.append(group_key)
                
                for col in self.df.columns:
                    if col not in new_data:
                        new_data[col] = []
                    new_data[col].append(self.df._data[col].data[first_idx])
        
        # Convertir listas a series
        for col in new_data:
            new_data[col] = MySeries(new_data[col], new_index, col)
        
        return MyDataFrame(new_data, new_index, self.df.columns)
    
    def reset_index(self):
        """
        Resetea el índice después de agrupar
        
        Retorna:
        - DataFrame con índice reseteado
        """
        result = self.first()
        return result.reset_index()

# ====================== MATPLOTLIB IMPLEMENTACIÓN ======================
class MyPyPlot:
    @staticmethod
    def figure(figsize=None, dpi=None, facecolor=None, edgecolor=None):
        """
        Crea una nueva figura
        
        Parámetros:
        - figsize: tamaño de la figura (ancho, alto) en pulgadas
        - dpi: resolución en puntos por pulgada
        - facecolor: color de fondo
        - edgecolor: color del borde
        
        Retorna:
        - Objeto figura
        """
        print(f"[Matplotlib] Creando figura: tamaño={figsize}, dpi={dpi}, facecolor={facecolor}, edgecolor={edgecolor}")
        return MyFigure()
    
    @staticmethod
    def gca():
        """
        Obtiene el eje actual
        
        Retorna:
        - Objeto eje
        """
        print("[Matplotlib] Obteniendo eje actual")
        return MyAxis()
    
    @staticmethod
    def plot(x, y=None, color=None, label=None):
        """
        Dibuja una línea
        
        Parámetros:
        - x: valores del eje x o valores y si y no se proporciona
        - y: valores del eje y
        - color: color de la línea
        - label: etiqueta para la leyenda
        
        Retorna:
        - Objeto línea
        """
        if y is None:
            y = x
            x = list(range(len(y)))
        
        print(f"[Matplotlib] Dibujando línea: color={color}, etiqueta={label}")
        if hasattr(y, 'data'):
            y_data = y.data  # Si es una serie de pandas
        else:
            y_data = y
            
        if len(y_data) > 5:
            print(f"[Matplotlib] Primeros 5 puntos: {list(zip(x[:5], y_data[:5]))}")
        else:
            print(f"[Matplotlib] Puntos: {list(zip(x, y_data))}")
        
        return MyLine()
    
    @staticmethod
    def title(text, fontsize=None):
        """
        Establece el título del gráfico
        
        Parámetros:
        - text: texto del título
        - fontsize: tamaño de la fuente
        """
        print(f"[Matplotlib] Título: {text}, tamaño={fontsize}")
    
    @staticmethod
    def xlabel(text, fontsize=None):
        """
        Establece la etiqueta del eje X
        
        Parámetros:
        - text: texto de la etiqueta
        - fontsize: tamaño de la fuente
        """
        print(f"[Matplotlib] Etiqueta X: {text}, tamaño={fontsize}")
    
    @staticmethod
    def ylabel(text, fontsize=None):
        """
        Establece la etiqueta del eje Y
        
        Parámetros:
        - text: texto de la etiqueta
        - fontsize: tamaño de la fuente
        """
        print(f"[Matplotlib] Etiqueta Y: {text}, tamaño={fontsize}")
    
    @staticmethod
    def legend(loc=None, prop=None):
        """
        Muestra la leyenda
        
        Parámetros:
        - loc: ubicación de la leyenda
        - prop: propiedades de la leyenda
        """
        print(f"[Matplotlib] Leyenda: posición={loc}, propiedades={prop}")
    
    @staticmethod
    def show():
        """Muestra el gráfico"""
        print("[Matplotlib] Mostrando gráfico...")

class MyFigure:
    """Clase simple para representar una figura"""
    pass

class MyAxis:
    """Clase para representar un eje"""
    def __init__(self):
        self.xaxis = MyAxisTicks()
        self.yaxis = MyAxisTicks()

class MyAxisTicks:
    """Clase para representar marcas en un eje"""
    def get_major_ticks(self):
        """
        Obtiene las marcas principales
        
        Retorna:
        - Lista de objetos marca
        """
        return [MyTick() for _ in range(5)]

class MyTick:
    """Clase para representar una marca en un eje"""
    def __init__(self):
        self.label1 = MyTickLabel()

class MyTickLabel:
    """Clase para representar la etiqueta de una marca"""
    def set_fontsize(self, size):
        """
        Establece el tamaño de la fuente
        
        Parámetros:
        - size: tamaño de la fuente
        """
        print(f"[Matplotlib] Estableciendo tamaño de fuente de marca: {size}")

class MyLine:
    """Clase simple para representar una línea"""
    pass

# ====================== SKLEARN IMPLEMENTACIÓN ======================
class MyRandomForestRegressor:
    def __init__(self, n_estimators=100, random_state=None):
        """
        Inicializa un modelo Random Forest para regresión
        
        Parámetros:
        - n_estimators: número de árboles
        - random_state: semilla para reproducibilidad
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self._X = None
        self._y = None
    
    def fit(self, X, y):
        """
        Entrena el modelo Random Forest
        
        Parámetros:
        - X: características de entrenamiento
        - y: etiquetas de entrenamiento
        
        Retorna:
        - self
        """
        if self.random_state is not None:
            random.seed(self.random_state)
        
        self._X = X
        self._y = y
        n_features = len(X[0]) if X and isinstance(X[0], list) else 1
        
        print(f"[RandomForest] Entrenando con {self.n_estimators} árboles y {n_features} características")
        
        # Simulamos la creación de árboles
        self.trees = []
        for i in range(self.n_estimators):
            # Cada árbol sería un modelo de decisión simple
            tree = MyDecisionTree(max_depth=random.randint(5, 15))
            
            # Bootstrap sampling (con reemplazo)
            indices = [random.randint(0, len(X)-1) for _ in range(len(X))]
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            
            # Entrenar árbol
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            if i % 100 == 0 and i > 0:
                print(f"[RandomForest] {i} árboles entrenados...")
        
        # Simular importancia de características
        self.feature_importances_ = [random.random() for _ in range(n_features)]
        total = sum(self.feature_importances_)
        self.feature_importances_ = [imp/total for imp in self.feature_importances_]
        
        print("[RandomForest] Entrenamiento completado")
        return self
    
    def predict(self, X):
        """
        Predice usando el modelo entrenado
        
        Parámetros:
        - X: características para predecir
        
        Retorna:
        - predicciones
        """
        if not self.trees:
            raise ValueError("El modelo no ha sido entrenado")
        
        print(f"[RandomForest] Prediciendo {len(X)} muestras")
        
        # Para cada muestra, cada árbol hace una predicción y luego se promedian
        predictions = []
        
        for sample in X:
            # Predicciones de todos los árboles
            tree_preds = [tree.predict_one(sample) for tree in self.trees]
            # Promedio de predicciones
            avg_pred = sum(tree_preds) / len(tree_preds)
            predictions.append(avg_pred)
        
        return predictions
    
    def score(self, X, y):
        """
        Calcula el coeficiente de determinación R^2
        
        Parámetros:
        - X: características
        - y: etiquetas reales
        
        Retorna:
        - R^2 score
        """
        predictions = self.predict(X)
        
        # Calcular R^2
        mean_y = sum(y) / len(y)
        ss_total = sum((yi - mean_y) ** 2 for yi in y)
        ss_residual = sum((yi - pred) ** 2 for yi, pred in zip(y, predictions))
        
        if ss_total == 0:
            return 0  # Para evitar división por cero
        
        r2 = 1 - (ss_residual / ss_total)
        return r2

class MyDecisionTree:
    def __init__(self, max_depth=10):
        """
        Inicializa un árbol de decisión
        
        Parámetros:
        - max_depth: profundidad máxima del árbol
        """
        self.max_depth = max_depth
        self.tree = None  # En una implementación real, aquí iría la estructura del árbol
        self._X = None
        self._y = None
    
    def fit(self, X, y):
        """
        Entrena el árbol de decisión
        
        Parámetros:
        - X: características de entrenamiento
        - y: etiquetas de entrenamiento
        
        Retorna:
        - self
        """
        self._X = X
        self._y = y
        
        # En una implementación real, aquí construiríamos el árbol
        # Para simplificar, solo almacenamos los datos
        
        return self
    
    def predict_one(self, sample):
        """
        Predice para una muestra
        
        Parámetros:
        - sample: características de la muestra
        
        Retorna:
        - predicción
        """
        # En una implementación real, recorreríamos el árbol
        # Para simplificar, usamos una heurística basada en los datos de entrenamiento
        
        # Encontrar las k muestras más cercanas
        k = min(5, len(self._X))
        distances = []
        
        for i, x in enumerate(self._X):
            # Distancia euclidiana
            dist = sum((a - b) ** 2 for a, b in zip(sample, x)) ** 0.5
            distances.append((dist, i))
        
        # Ordenar por distancia
        distances.sort()
        
        # Promedio de las k muestras más cercanas
        nearest_indices = [idx for _, idx in distances[:k]]
        nearest_y = [self._y[idx] for idx in nearest_indices]
        
        return sum(nearest_y) / len(nearest_y)

# ====================== FUNCIONES DE UTILIDAD ======================
def Grouper(key, freq):
    """
    Crea un objeto agrupador para pandas
    
    Parámetros:
    - key: columna a agrupar
    - freq: frecuencia
    
    Retorna:
    - objeto agrupador
    """
    return MyGrouper(key, freq)

# ====================== EXPORTAR COMO MÓDULOS ======================
np = MyNumpy()
pd = type('', (), {
    'read_csv': MyDataFrame.read_csv,
    'DataFrame': MyDataFrame,
    'Series': MySeries,
    'Grouper': Grouper
})
plt = MyPyPlot()
train_test_split = lambda X, y, test_size=0.2, random_state=None: (X[:int(len(X)*(1-test_size))], X[int(len(X)*(1-test_size)):], y[:int(len(y)*(1-test_size))], y[int(len(y)*(1-test_size)):])
RandomForestRegressor = MyRandomForestRegressor