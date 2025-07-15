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
    
    def __len__(self):
        """
        Devuelve la longitud de la serie
        
        Retorna:
        - Número de elementos en la serie
        """
        return len(self.data)
    
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

    def __len__(self):
        """Devuelve la longitud del DataFrame"""
        return len(self.index)
    
    def __setitem__(self, key, value):
        """
        Permite asignación de columnas usando df[columna] = valor
        
        Parámetros:
        - key: nombre de la columna
        - value: MySeries o lista de valores
        """
        if isinstance(value, MySeries):
            # Si es una MySeries, usarla directamente
            self._data[key] = value
        elif isinstance(value, list):
            # Si es una lista, convertir a MySeries
            self._data[key] = MySeries(value, self.index, key)
        else:
            # Para valores escalares, crear una serie con el mismo valor
            scalar_data = [value] * len(self.index)
            self._data[key] = MySeries(scalar_data, self.index, key)
        
        # Actualizar la lista de columnas si es una columna nueva
        if key not in self.columns:
            self.columns.append(key)

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
    
    def copy(self):
        """
        Crea una copia del DataFrame
        
        Retorna:
        - Nueva instancia de MyDataFrame
        """
        new_data = {}
        for col, series in self._data.items():
            new_data[col] = MySeries(series.data.copy(), series.index.copy(), col)
        
        return MyDataFrame(new_data, self.index.copy(), self.columns.copy())

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
        
        # Filtrar solo columnas que existen en _data
        valid_columns = []
        for col in self.columns:
            if col in self._data and self._data[col] is not None:
                try:
                    new_data[col] = MySeries(self._data[col].data, new_index, col)
                    valid_columns.append(col)
                except Exception as e:
                    print(f"[Warning] Error procesando columna '{col}': {e}")
            else:
                print(f"[Warning] Columna '{col}' no encontrada en _data, omitiendo...")
        
        # Actualizar la lista de columnas válidas
        self.columns = valid_columns
        
        # Añadir el índice anterior como columna si no es numérico secuencial
        if (self.index and 
            not all(isinstance(idx, int) and idx == i for i, idx in enumerate(self.index))):
            new_data['index'] = MySeries(self.index, new_index, 'index')
            valid_columns = ['index'] + valid_columns
        
        return MyDataFrame(new_data, new_index, valid_columns)

    
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
    
    def groupby(self, by):
        """
        Agrupa el DataFrame por una o más columnas
        """
        print(f"[DEBUG] Iniciando groupby con: {by}")
        print(f"[DEBUG] Columnas disponibles antes de groupby: {self.columns}")
        print(f"[DEBUG] Datos disponibles: {list(self._data.keys())}")
        
        return MyGroupBy(self, by)
    
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
        Comprueba valores nulos y devuelve un DataFrame compatible
        """
        null_data = {}
        
        for col in self.columns:
            if col in self._data:
                null_values = []
                for val in self._data[col].data:
                    is_null = val is None or (isinstance(val, float) and math.isnan(val))
                    null_values.append(is_null)
                null_data[col] = MySeries(null_values, self.index, col)
        
        result_df = MyDataFrame(null_data, self.index, list(null_data.keys()))
        
        # Crear una clase que extienda el DataFrame con método any()
        class DataFrameWithAny(MyDataFrame):
            def __init__(self, data, index, columns):
                super().__init__(data, index, columns)
            
            @property
            def values(self):
                """
                Devuelve los valores como una estructura compatible con NumPy
                """
                if not self._data or not self.columns:
                    return []
                
                result = []
                for i in range(len(self.index)):
                    row = []
                    for col in self.columns:
                        if col in self._data and i < len(self._data[col].data):
                            row.append(self._data[col].data[i])
                        else:
                            row.append(None)
                    result.append(row)
                
                # Crear una clase que simule el comportamiento de NumPy array
                class ArrayLike:
                    def __init__(self, data):
                        self.data = data
                    
                    def __getitem__(self, key):
                        if isinstance(key, tuple):
                            # Manejo de indexado multidimensional
                            row_slice, col_slice = key
                            
                            # Aplicar slice de filas
                            if isinstance(row_slice, slice):
                                start = row_slice.start or 0
                                stop = row_slice.stop or len(self.data)
                                step = row_slice.step or 1
                                row_indices = list(range(start, stop, step))
                            else:
                                row_indices = [row_slice] if isinstance(row_slice, int) else row_slice
                            
                            # Aplicar slice de columnas
                            if isinstance(col_slice, slice):
                                start = col_slice.start or 0
                                stop = col_slice.stop or len(self.data[0]) if self.data else 0
                                step = col_slice.step or 1
                                col_indices = list(range(start, stop, step))
                            else:
                                col_indices = [col_slice] if isinstance(col_slice, int) else col_slice
                            
                            # Extraer datos
                            result = []
                            for row_idx in row_indices:
                                if row_idx < len(self.data):
                                    row = []
                                    for col_idx in col_indices:
                                        if col_idx < len(self.data[row_idx]):
                                            row.append(self.data[row_idx][col_idx])
                                        else:
                                            row.append(None)
                                    result.append(row)
                            
                            return result
                        else:
                            # Indexado simple
                            return self.data[key]
                    
                    def __len__(self):
                        return len(self.data)
                    
                    def __iter__(self):
                        return iter(self.data)
                
                return ArrayLike(result)

    
    def debug_structure(self):
        """
        Método de depuración para verificar la estructura del DataFrame
        """
        print(f"Columnas declaradas: {self.columns}")
        print(f"Columnas en _data: {list(self._data.keys())}")
        print(f"Longitud del índice: {len(self.index)}")
        
        # Verificar inconsistencias
        missing_in_data = [col for col in self.columns if col not in self._data]
        extra_in_data = [col for col in self._data.keys() if col not in self.columns]
        
        if missing_in_data:
            print(f"⚠️  Columnas faltantes en _data: {missing_in_data}")
        if extra_in_data:
            print(f"ℹ️  Columnas extra en _data: {extra_in_data}")

    
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
        """
        data = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                print(f"[DEBUG] Headers encontrados: {headers}")
                
                # Limpiar headers
                clean_headers = [header.strip() for header in headers]
                
                # Inicializar columnas
                for header in clean_headers:
                    data[header] = []
                
                # Leer datos fila por fila
                row_count = 0
                for row in reader:
                    row_count += 1
                    for i, value in enumerate(row):
                        if i < len(clean_headers):
                            header = clean_headers[i]
                            
                            # Convertir fechas si es necesario
                            if parse_dates and i in parse_dates and date_parser:
                                try:
                                    parsed_value = date_parser(value)
                                    data[header].append(parsed_value)
                                except Exception as e:
                                    print(f"[WARNING] Error parseando fecha '{value}': {e}")
                                    data[header].append(None)
                            else:
                                # Procesar valores
                                if value.strip() == '' or value.strip().lower() == 'nan':
                                    data[header].append(None)
                                else:
                                    try:
                                        # Intentar convertir a número
                                        if '.' in value or 'e' in value.lower():
                                            numeric_value = float(value)
                                        else:
                                            numeric_value = int(value)
                                        data[header].append(numeric_value)
                                    except ValueError:
                                        # Mantener como string
                                        data[header].append(value.strip())
                
                print(f"[DEBUG] Filas procesadas: {row_count}")
                print(f"[DEBUG] Columnas con datos: {list(data.keys())}")
                
                # Verificar que hay datos
                if not data or all(len(values) == 0 for values in data.values()):
                    print("[ERROR] No se encontraron datos en el archivo")
                    return MyDataFrame()
                
                # Crear DataFrame
                return MyDataFrame(data, columns=clean_headers)
        
        except Exception as e:
            print(f"[ERROR] Error al leer el archivo CSV: {e}")
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
        Inicializa un objeto GroupBy con mejor manejo de datos
        """
        self.df = df
        self.by = by
        self.groups = self._create_groups()

        print(f"[DEBUG] MyGroupBy inicializado con {len(self.groups)} grupos")
    
    def _create_groups(self):
        """Crea los grupos según la columna especificada"""
        groups = defaultdict(list)
        
        if isinstance(self.by, list) and len(self.by) == 1 and isinstance(self.by[0], MyGrouper):
            grouper = self.by[0]
            key_col = grouper.key
            
            print(f"[DEBUG] Agrupando por columna: {key_col}")
            
            if key_col in self.df._data:
                if grouper.freq == 'D':
                    for i, timestamp in enumerate(self.df._data[key_col].data):
                        if timestamp is not None:
                            try:
                                if hasattr(timestamp, 'date'):
                                    date_key = timestamp.date()
                                elif isinstance(timestamp, datetime.datetime):
                                    date_key = timestamp.date()
                                else:
                                    # Intentar parsear string
                                    if isinstance(timestamp, str):
                                        dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d')
                                        date_key = dt.date()
                                    else:
                                        continue
                                
                                groups[date_key].append(i)
                            except Exception as e:
                                print(f"[WARNING] Error procesando timestamp {timestamp}: {e}")
                                continue
            else:
                print(f"[WARNING] Columna '{key_col}' no encontrada para agrupación")
        
        print(f"[DEBUG] Se crearon {len(groups)} grupos")
        return groups

    
    def first(self):
        """
        Devuelve el primer valor de cada grupo
        """
        if not self.groups:
            print("[WARNING] No hay grupos para procesar")
            return MyDataFrame()
        
        new_data = {}
        new_index = []
        
        # Inicializar columnas basándose en datos reales
        for col in self.df._data.keys():
            new_data[col] = []
        
        for group_key, indices in self.groups.items():
            if indices:
                first_idx = indices[0]
                new_index.append(group_key)
                
                for col in self.df._data.keys():
                    if first_idx < len(self.df._data[col].data):
                        new_data[col].append(self.df._data[col].data[first_idx])
                    else:
                        new_data[col].append(None)
        
        # Convertir listas a series
        for col in new_data:
            new_data[col] = MySeries(new_data[col], new_index, col)
        
        result = MyDataFrame(new_data, new_index, list(new_data.keys()))
        print(f"[DEBUG] Resultado de first(): {result.shape} con columnas {result.columns}")
        
        return result
    
    def reset_index(self):
        """
        Resetea el índice después de agrupar
        """
        result = self.first()
        if result.shape[0] > 0:
            return result.reset_index()
        else:
            print("[WARNING] No hay datos para resetear índice")
            return result

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
        Dibuja una línea (versión mejorada)
        
        Parámetros:
        - x: valores del eje x o valores y si y no se proporciona
        - y: valores del eje y
        - color: color de la línea
        - label: etiqueta para la leyenda
        """
        if y is None:
            y = x
            x = list(range(len(y)))
        
        # Extraer datos de MySeries si es necesario
        if hasattr(y, 'data'):
            y_data = y.data
        else:
            y_data = y
        
        if hasattr(x, 'data'):
            x_data = x.data
        else:
            x_data = x
        
        print(f"[Matplotlib] Dibujando línea: color={color}, etiqueta={label}")
        
        # Verificar que tenemos datos
        if len(y_data) == 0:
            print("[Matplotlib] No hay datos para graficar")
            return MyLine()
        
        # Mostrar información de los datos
        if len(y_data) > 5:
            print(f"[Matplotlib] Primeros 5 puntos: {list(zip(x_data[:5], y_data[:5]))}")
        else:
            print(f"[Matplotlib] Puntos: {list(zip(x_data, y_data))}")
        
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
        Calcula el coeficiente de determinación R^2 (versión corregida)
        
        Parámetros:
        - X: características
        - y: etiquetas reales
        
        Retorna:
        - R^2 score
        """
        predictions = self.predict(X)
        
        # Aplanar y para manejar listas de listas
        y_flat = []
        for item in y:
            if isinstance(item, list):
                y_flat.extend(item)  # Si es lista, extraer elementos
            else:
                y_flat.append(item)  # Si es número, agregar directamente
        
        # Verificar que tenemos el mismo número de predicciones y valores reales
        if len(predictions) != len(y_flat):
            print(f"[WARNING] Longitudes diferentes: predicciones={len(predictions)}, y_real={len(y_flat)}")
            min_len = min(len(predictions), len(y_flat))
            predictions = predictions[:min_len]
            y_flat = y_flat[:min_len]
        
        # Calcular R^2
        if len(y_flat) == 0:
            return 0
        
        mean_y = sum(y_flat) / len(y_flat)
        ss_total = sum((yi - mean_y) ** 2 for yi in y_flat)
        ss_residual = sum((yi - pred) ** 2 for yi, pred in zip(y_flat, predictions))
        
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
            # Calcular distancia con manejo seguro de tipos
            dist = 0
            for a, b in zip(sample, x):
                try:
                    # Para tipos numéricos, usar distancia euclidiana normal
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        dist += (a - b) ** 2
                    # Para objetos datetime, usar la diferencia en segundos
                    elif isinstance(a, datetime.datetime) and isinstance(b, datetime.datetime):
                        diff_seconds = abs((a - b).total_seconds())
                        dist += diff_seconds ** 2
                    # Para objetos date, convertir a datetime y calcular diferencia
                    elif isinstance(a, datetime.date) and isinstance(b, datetime.date):
                        a_dt = datetime.datetime.combine(a, datetime.time.min)
                        b_dt = datetime.datetime.combine(b, datetime.time.min)
                        diff_seconds = abs((a_dt - b_dt).total_seconds())
                        dist += diff_seconds ** 2
                    # Para otros tipos, usar una distancia binaria (iguales o diferentes)
                    else:
                        dist += 0 if a == b else 1
                except Exception as e:
                    # En caso de error, usar una distancia predeterminada alta
                    print(f"[WARNING] Error calculando distancia entre {type(a)} y {type(b)}: {e}")
                    dist += 100  # Valor arbitrario alto
            
            dist = dist ** 0.5  # Raíz cuadrada para completar la distancia euclidiana
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
