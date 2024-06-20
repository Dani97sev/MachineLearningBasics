import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leemos el dataframe
df = pd.read_csv('../50_Startups.csv')

# Asignamos los indices a X e Y
# Seleccionamos todas las columnas excepto la última como características (X)
X = df.iloc[:,:-1]
# Seleccionamos la última columna como vector de etiquetas (Y) y la convertimos en un array
Y = df.iloc[:,-1].values

# Esta linea sirve para mostrar el numero de filas que uno quiera
print(df.head(7))

'''X: Esta línea toma todas las filas (:) y todas las columnas excepto la última (:-1) del DataFrame df. 
Con la funcion iloc seleccionamos partes específicas de un DataFrame basadas en posiciones numéricas de índice. 
En el contexto de aprendizaje automático, esto corresponde a las variables independientes que se usarán para el entrenamiento del modelo.

Y: seleccionamos todas las filas y solo la última columna (-1). El método .values convierte los valores del DataFrame seleccionado en un array.
En muchos algoritmos de aprendizaje automático, es útil o necesario trabajar con arrays en lugar de DataFrames de pandas, 
especialmente para las etiquetas o variables dependientes, que en este caso, se espera que sean unidimensionales.'''

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Configuramos el transformador para aplicar codificación One-Hot a la columna de índice 3 (que es la que nos interesa, en este caso),ç
# manteniendo el resto de columnas sin cambios (con el passthrough)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')

# Aplicamos la transformación a las características "X" y convertimos el resultado en un array
X = np.array(ct.fit_transform(X))

# Quita el comentario de este print si quieres ver si la codificacion se ha realizado correctamente
# print(X)

from sklearn.model_selection import train_test_split
# Usamos la funcion train_test_split para dividir los datos. Le damos tamaño de prueba, subrayado y el valor de 0.2,
# lo que indica que el 20% de los datos seran el conjunto de pruebas y los restantes seran del conjunto de entrenamiento
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)


# Importamos la clase LinearRegression que aplicaremos a nuestro conjunto de entrenamiento para,
# acto seguido, ajustar los datos de entrenamiento X_train e Y_train para que se produzca el proceso de entrenamiento 
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, Y_train)


# Usamos "predict" para predecir los valores de nuestros datos de prueba X_test
Y_pred = reg.predict(X_test)

# Mostramos ahora, en un marco de datos pandas, los valores reales y los valores predichos
df = pd.DataFrame({'Valores reales': Y_test, 'Valores predichos': Y_pred})
print(df)

'''Podemos comprobar que los valores precichos se acercan bastante a los valores reales, por lo tanto, el modelo tiene buena precision,
pero ¿Cómo de buena es esta precision?'''

# Vamos a usar el error cuadratico medio (RMSE) para evaluar nuestra precision
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# Podriamos decir que el modelo de regresion en este caso es bueno,
# dado el valor del error y la amplitud de variacion de los valores de la columna "Profit"