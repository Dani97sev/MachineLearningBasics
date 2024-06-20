import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Carga de datos desde un csv
df = pd.read_csv("../user+data.csv")

# Seleccionar columnas específicas para las características y la variable objetivo
# Extrae los valores de las columnas en las posiciones 2 y 4 como características
X = df.iloc[:, 2:4].values
# Extrae los valores de la columna en la posición 4 como variable objetivo
Y = df.iloc[:,4].values

#print(X) # Imprimir las características para verificar
#print(Y) # Imprimir la variable objetivo para verificar

from sklearn.model_selection import train_test_split

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=0)

from sklearn.preprocessing import StandardScaler

# Escalar las características para normalizar la escala de los datos
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Descomentar para imprimir los datos escalados
# print(X_train)
# print(X_test)

from sklearn.linear_model import LogisticRegression

# Crear y entrenar el modelo de regresión logística
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

# Realizar predicciones en el conjunto de prueba
Y_pred = classifier.predict(X_test)
print(Y_pred)

import seaborn as sns

# Calcular la matriz de confusión para evaluar el rendimiento del modelo
cm = confusion_matrix(Y_pred,Y_test)
print(cm)

from sklearn import metrics
print("Accuracy", metrics.accuracy_score(Y_test,Y_pred))

# Visualizar la matriz de confusión como un mapa de calor
sns.heatmap(cm)
# Mostrar el mapa de calor, si no se visualiza con la funcion de arriba
plt.show()