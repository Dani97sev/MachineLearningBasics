import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''Realizamos la prediccion del precio de la vivienda usando un modelo de regresion lineal simple'''

#Cargamos los datos del csv en una variable para poder manejarlos

df = pd.read_csv("../homeprices.csv")
print(df)

# Trazamos los puntos de dispersion de los datos para tener una idea de la distribucion de los mismos

plt.xlabel('Area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(df.Area, df.Price, color = 'red', marker='+')

# Importante para poder ver la gráfica en entornos de programacion como VSCode (quitar la almohadilla, si se quiere ver la grafica)
# plt.show()

'''Despues de ver la grafica, comprobamos que la distribucion de los puntos 
es adecuada para el uso de un modelo de regresion lineal, ya que los puntos no estan distribuidos al azar
y se muestran hacia la derecha y hacia arriba'''

# Uso del modelo de regresion lineal
# Creamos un objeto de regresion lineal
reg = LinearRegression()


# Ajustaremos los datos
reg.fit(df[['Area']], df.Price)


# Realizamos una prediccion con los datos ajustados
prediccion = reg.predict(pd.DataFrame([[3300]], columns=['Area']))
print(prediccion)

'''Como sabemos, la formula de la recta es y = m*x + b, donde "y" es el precio que queremos predecir,
"m" y "b" son los coeficientes que usaremos para hacer la regresion'''

m = reg.coef_
b = reg.intercept_
print("El valor de m es: ", m)
print("El valor de b es: ", b)

# Si usamos estos dos valores para sustituirlos en la funcion de la recta, nos quedaría:

x = 3300
y = (m*x) + b

print("El valor de la casa usando los coeficientes es: ", y)

'''Si ejecutamos este codigo, podemos ver que, tanto usando reg.precit como usando los coeficientes 
y sustiyendolos en la funcion, nos sale el mismo valor de la predicción'''