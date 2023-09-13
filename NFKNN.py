# Listas para almacenar los datos y etiquetas
dataFeat = []
dataLabel = []

# Inicializar listas para los datos de entrenamiento y etiquetas de entrenamiento
datos_entrenamiento = []
etiquetas_entrenamiento = []

# Inicializar listas para los datos de prueba, validación y etiquetas correspondientes
datos_prueba = []
etiquetas_prueba = []
datos_validacion = []
etiquetas_validacion = []

# Proporciones para dividir los datos
train_ratio = 0.6  # 60% para entrenamiento
validation_ratio = 0.2  # 20% para validación
test_ratio = 0.2  # 20% para prueba

# Abrir el archivo CSV para lectura
with open('Iris.csv', 'r') as archivo:
    # Saltar la fila de encabezado
    next(archivo)
    
    # Iterar a través de cada línea en el archivo
    for linea in archivo:
        # Dividir la línea en valores individuales utilizando la coma como delimitador
        valores = linea.strip().split(',')
        
        # El último valor en cada línea es la etiqueta (por ejemplo, Iris-setosa)
        etiqueta = valores[-1]
        
        # Los valores restantes son las características, también eliminamos el ID
        caracteristicas = [float(valor) for valor in valores[1:-1]]

        # Agregar las características y etiquetas a las listas correspondientes
        dataFeat.append(caracteristicas)
        dataLabel.append(etiqueta)

# Determinar el número de muestras para cada conjunto
total_samples = len(dataFeat)
num_train = int(train_ratio * total_samples)
num_validation = int(validation_ratio * total_samples)
num_test = total_samples - num_train - num_validation

# Crear una lista de índices
indices = list(range(total_samples))

# Barajar los índices personalmente sin usar la biblioteca random
for i in range(len(indices) - 1, 0, -1):
    # Generar un índice pseudoaleatorio sin usar la biblioteca random
    j = (i * 997 + 173) % len(indices)
    
    # Intercambiar elementos en los índices i y j
    indices[i], indices[j] = indices[j], indices[i]

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_indices = indices[:num_train]
validation_indices = indices[num_train:num_train + num_validation]
test_indices = indices[num_train + num_validation:]

# Llenar las listas de datos y etiquetas para cada conjunto
datos_entrenamiento = [dataFeat[i] for i in train_indices]
etiquetas_entrenamiento = [dataLabel[i] for i in train_indices]

datos_validacion = [dataFeat[i] for i in validation_indices]
etiquetas_validacion = [dataLabel[i] for i in validation_indices]

datos_prueba = [dataFeat[i] for i in test_indices]
etiquetas_prueba = [dataLabel[i] for i in test_indices]

# Imprimir el tamaño de los conjuntos de datos
print(f"Datos de entrenamiento: {len(datos_entrenamiento)} muestras")
print(f"Datos de validación: {len(datos_validacion)} muestras")
print(f"Datos de prueba: {len(datos_prueba)} muestras")

# Ahora, 'datos_entrenamiento' contiene una lista de vectores de características y
# 'etiquetas_entrenamiento' contiene una lista de etiquetas correspondientes para entrenamiento.
# 'datos_prueba' contiene una lista de vectores de características y 'etiquetas_prueba'
# contiene una lista de etiquetas correspondientes para pruebas.
# 'datos_validacion' contiene una lista de vectores de características y 'etiquetas_validacion'
# contiene una lista de etiquetas correspondientes para validación.

# Definir las funciones KNN

#Calcula la distancia Euclidiana entre dos puntos.
def euclidean_distance(punto1, punto2):
    distancia_cuadrada = sum((a - b) ** 2 for a, b in zip(punto1, punto2))
    return distancia_cuadrada ** 0.5

#Calcula la distancia de Manhattan entre dos puntos.
def manhattan_distance(punto1, punto2):
    return sum(abs(a - b) for a, b in zip(punto1, punto2))

# Calcula distancias entre un punto de prueba y todos los puntos de entrenamiento.
def calculate_distances(punto_prueba, training_data, metric='euclidean'):
    distancias = []
    for punto in training_data:
        if metric == 'euclidean':
            distancia = euclidean_distance(punto_prueba, punto)
        elif metric == 'manhattan':
            distancia = manhattan_distance(punto_prueba, punto)
        else:
            raise ValueError("Métrica de distancia no soportada. Las métricas soportadas son 'euclidean' y 'manhattan'.")
        distancias.append(distancia)
    return distancias

# Encuentra los índices de los k vecinos más cercanos basados en las distancias.
def k_indices(distancias, k):
    indices = list(range(len(distancias)))
    indices.sort(key=lambda i: distancias[i])
    vecinos_mas_cercanos_indices = indices[:k]
    return vecinos_mas_cercanos_indices

#Predice la etiqueta para un punto de prueba usando vecinos más cercanos (KNN).
def knn_predict(punto_prueba, training_data, training_labels, k, metric='euclidean'):
    distancias = calculate_distances(punto_prueba, training_data, metric=metric)
    vecinos_cercanos_indices = k_indices(distancias, k)
    etiquetas_vecinos = [training_labels[i] for i in vecinos_cercanos_indices]
    etiqueta_predicha = max(set(etiquetas_vecinos), key=etiquetas_vecinos.count)
    return etiqueta_predicha

# Nuevo punto de prueba
nuevo_punto_prueba = [5.1, 3.5, 1.4, 0.2]  # Reemplaza esto con tu punto de prueba

# Número de vecinos a considerar
k = 3

# Predecir la etiqueta para el nuevo punto de prueba usando el clasificador KNN
etiqueta_predicha = knn_predict(nuevo_punto_prueba, datos_entrenamiento, etiquetas_entrenamiento, k, metric='euclidean')
print("Etiqueta Predicha:", etiqueta_predicha)

# EVALUACIÓN DEL CONJUNTO DE ENTRENAMIENTO

# Inicializar listas para almacenar las predicciones
predictionsTrain = []

# Calcular predicciones para todos los puntos de datos de entrenamiento
for punto_Train in datos_entrenamiento:
    etiqueta_predicha = knn_predict(punto_Train, datos_entrenamiento, etiquetas_entrenamiento, k, metric='euclidean')
    predictionsTrain.append(etiqueta_predicha)

# Calcular la precisión
predicciones_correctas = sum(1 for p, a in zip(predictionsTrain, etiquetas_entrenamiento) if p == a)
total_predicciones = len(etiquetas_entrenamiento)
precision = predicciones_correctas / total_predicciones
print("Precisión:", precision)

# Inicializar un conjunto vacío para almacenar etiquetas únicas
etiquetas_unicas = set()

# Iterar a través de la lista de etiquetas de datos para agregar etiquetas únicas al conjunto
for etiqueta in dataLabel:
    etiquetas_unicas.add(etiqueta)

# Convertir el conjunto de nuevo a una lista si es necesario
etiquetas_unicas_lista = list(etiquetas_unicas)

# Crear un diccionario para mapear etiquetas a índices
etiqueta_a_indice = {etiqueta: i for i, etiqueta in enumerate(etiquetas_unicas_lista)}

# Inicializar una matriz de confusión con ceros
matriz_confusionEntrenamiento = [[0 for _ in etiquetas_unicas_lista] for _ in etiquetas_unicas_lista]

# Iterar a través de todas las predicciones y etiquetas reales para llenar la matriz de confusión
for predicha, real in zip(predictionsTrain, etiquetas_entrenamiento):
    indice_predicho = etiqueta_a_indice[predicha]
    indice_real = etiqueta_a_indice[real]
    matriz_confusionEntrenamiento[indice_real][indice_predicho] += 1

# Mostrar la matriz de confusión
print("Matriz de Confusión Entrenamiento:")
for i in range(len(etiquetas_unicas_lista)):
    for j in range(len(etiquetas_unicas_lista)):
        print(matriz_confusionEntrenamiento[i][j], end="\t")
    print()

# Calcular la recuperación (recall) para cada clase
puntajes_recall = []
for i in range(len(etiquetas_unicas_lista)):
    verdadero_positivo = matriz_confusionEntrenamiento[i][i]
    positivo_real = sum(matriz_confusionEntrenamiento[i])
    recuperacion = verdadero_positivo / positivo_real if positivo_real > 0 else 0
    puntajes_recall.append(recuperacion)

# Calcular el F1 Score para cada clase
puntajes_f1 = []
for i in range(len(etiquetas_unicas_lista)):
    precision = matriz_confusionEntrenamiento[i][i] / sum(matriz_confusionEntrenamiento[j][i] for j in range(len(etiquetas_unicas_lista)))
    recuperacion = puntajes_recall[i]
    f1 = 2 * (precision * recuperacion) / (precision + recuperacion) if precision + recuperacion > 0 else 0
    puntajes_f1.append(f1)

# Imprimir la recuperación y el F1 Score para cada clase
for i, etiqueta in enumerate(etiquetas_unicas_lista):
    print(f"Clase '{etiqueta}' - Recuperación: {puntajes_recall[i]:.2f}, F1 Score: {puntajes_f1[i]:.2f}")

# EVALUACIÓN DEL CONJUNTO DE PRUEBA

# Inicializar listas para almacenar las predicciones
predictionsPrueb = []

# Calcular predicciones para todos los puntos de datos de prueba
for punto_prueba in datos_prueba:
    etiqueta_predicha = knn_predict(punto_prueba, datos_prueba, etiquetas_prueba, k, metric='euclidean')
    predictionsPrueb.append(etiqueta_predicha)

# Calcular la precisión
predicciones_correctas = sum(1 for p, a in zip(predictionsPrueb, etiquetas_prueba) if p == a)
total_predicciones = len(etiquetas_prueba)
precision = predicciones_correctas / total_predicciones
print("Precisión:", precision)

# Inicializar una matriz de confusión con ceros
matriz_confusionPrueba = [[0 for _ in etiquetas_unicas_lista] for _ in etiquetas_unicas_lista]

# Iterar a través de todas las predicciones y etiquetas reales para llenar la matriz de confusión
for predicha, real in zip(predictionsPrueb, etiquetas_prueba):
    indice_predicho = etiqueta_a_indice[predicha]
    indice_real = etiqueta_a_indice[real]
    matriz_confusionPrueba[indice_real][indice_predicho] += 1

# Mostrar la matriz de confusión
print("Matriz de Confusión Prueba:")
for i in range(len(etiquetas_unicas_lista)):
    for j in range(len(etiquetas_unicas_lista)):
        print(matriz_confusionPrueba[i][j], end="\t")
    print()

# Calcular la recuperación (recall) para cada clase
puntajes_recall = []
for i in range(len(etiquetas_unicas_lista)):
    verdadero_positivo = matriz_confusionPrueba[i][i]
    positivo_real = sum(matriz_confusionPrueba[i])
    recuperacion = verdadero_positivo / positivo_real if positivo_real > 0 else 0
    puntajes_recall.append(recuperacion)

# Calcular el F1 Score para cada clase
puntajes_f1 = []
for i in range(len(etiquetas_unicas_lista)):
    precision = matriz_confusionPrueba[i][i] / sum(matriz_confusionPrueba[j][i] for j in range(len(etiquetas_unicas_lista)))
    recuperacion = puntajes_recall[i]
    f1 = 2 * (precision * recuperacion) / (precision + recuperacion) if precision + recuperacion > 0 else 0
    puntajes_f1.append(f1)

# Imprimir la recuperación y el F1 Score para cada clase
for i, etiqueta in enumerate(etiquetas_unicas_lista):
    print(f"Clase '{etiqueta}' - Recuperación: {puntajes_recall[i]:.2f}, F1 Score: {puntajes_f1[i]:.2f}")

# EVALUACIÓN DEL CONJUNTO DE VALIDACIÓN

# Inicializar listas para almacenar las predicciones
predictionsVal = []

# Calcular predicciones para todos los puntos de datos de validación
for punto_val in datos_validacion:
    etiqueta_predicha = knn_predict(punto_val, datos_validacion, etiquetas_validacion, k, metric='euclidean')
    predictionsVal.append(etiqueta_predicha)

# Calcular la precisión
predicciones_correctas = sum(1 for p, a in zip(predictionsVal, etiquetas_validacion) if p == a)
total_predicciones = len(etiquetas_validacion)
precision = predicciones_correctas / total_predicciones
print("Precisión:", precision)

# Inicializar una matriz de confusión con ceros
matriz_confusionVal = [[0 for _ in etiquetas_unicas_lista] for _ in etiquetas_unicas_lista]

# Iterar a través de todas las predicciones y etiquetas reales para llenar la matriz de confusión
for predicha, real in zip(predictionsVal, etiquetas_validacion):
    indice_predicho = etiqueta_a_indice[predicha]
    indice_real = etiqueta_a_indice[real]
    matriz_confusionVal[indice_real][indice_predicho] += 1

# Mostrar la matriz de confusión
print("Matriz de Confusión Validación:")
for i in range(len(etiquetas_unicas_lista)):
    for j in range(len(etiquetas_unicas_lista)):
        print(matriz_confusionVal[i][j], end="\t")
    print()

# Calcular la recuperación (recall) para cada clase
puntajes_recall = []
for i in range(len(etiquetas_unicas_lista)):
    verdadero_positivo = matriz_confusionVal[i][i]
    positivo_real = sum(matriz_confusionVal[i])
    recuperacion = verdadero_positivo / positivo_real if positivo_real > 0 else 0
    puntajes_recall.append(recuperacion)

# Calcular el F1 Score para cada clase
puntajes_f1 = []
for i in range(len(etiquetas_unicas_lista)):
    precision = matriz_confusionVal[i][i] / sum(matriz_confusionVal[j][i] for j in range(len(etiquetas_unicas_lista)))
    recuperacion = puntajes_recall[i]
    f1 = 2 * (precision * recuperacion) / (precision + recuperacion) if precision + recuperacion > 0 else 0
    puntajes_f1.append(f1)

# Imprimir la recuperación y el F1 Score para cada clase
for i, etiqueta in enumerate(etiquetas_unicas_lista):
    print(f"Clase '{etiqueta}' - Recuperación: {puntajes_recall[i]:.2f}, F1 Score: {puntajes_f1[i]:.2f}")

