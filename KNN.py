import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


df = pd.read_csv("../user+data.csv")
X = df.iloc[:, 2:4].values
Y = df.iloc[:, 4].values

# print(X)
# print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

# Escalamos nuestros datos para normalizarlos en un rango especifico

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_test)

# Como necesitamos ajustar el clasificador KNN a los datos de entrenamiento
from sklearn.neighbors import KNeighborsClassifier

# Despues de importar el clasificador hay que hacer los siguiente:
# 1. Crear un objeto clasificador de la clase
#  1.1 El parametro n_neighbors define los vecinos necesarios para el algoritmo
#  1.2 El parametro metric, en concreto con el valor minkowski, es por defecto y establece la distancia entre los puntos
#  1.3 El parametro p con el valor 2 es equivalente a la metrica euclidea estandar
classifier = KNeighborsClassifier(n_neighbors=5,metric="minkowski", p=2)

# Ajustamos el clasificador de nuestro conjunto de datos de entrenamiento
# Con esto conseguiremos que se clasifiquen las tendencias y predice la salida, basandose en los datos de entrenamiento
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)


# Calcula la matriz de confusión comparando las etiquetas predichas (Y_pred) con las etiquetas reales (Y_test)
cm = confusion_matrix(Y_pred, Y_test)

# Imprime la matriz de confusión para visualizar el rendimiento del clasificador
# La matriz muestra cuántas predicciones fueron correctas e incorrectas para cada clase
print(cm)

from sklearn import metrics
print("Accuracy", metrics.accuracy_score(Y_test,Y_pred))

'''La salida de esta funcion es la siguiente:
[[64  3]
 [ 4 29]]
 
 ¿Que conclusiones sacamos de esto?: 
 1. 64 valores verdaderos se predicen correctamente verdaderos y 3 valores falsos se predicen incorrectamente verdaderos
 2. 4 valores verdaderos se predicen incorrectamente falsos y 29 valores falsos se predicen correctamente falsos
 
 Con lo cual tenemos que:
 - 64+29 = 93 predicciones correctas
 - 3+4 = 7 predicciones incorrectas
'''