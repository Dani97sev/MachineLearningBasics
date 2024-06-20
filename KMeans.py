import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Configuración del tamaño de la figura y estilo de los gráficos
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')


# Cargar los datos del archivo CSV (en este caso, .txt)
df = pd.read_csv("../mallCustomerData.txt", sep = ',')
print(df.shape) # Imprimir número de filas y columnas del Data Frame
print(df.head(10)) # Imprimir las primeras 10 filas del DataFrame


# Contar los valores en la columna 'Gender'
print(df['Gender'].value_counts())


# Obtener los valores de las columnas 'Annual Income (k$)' y 'Spending Score (1-100)'
f1 = df['Annual Income (k$)'].values
f2 = df['Spending Score (1-100)'].values


# Imprimir las claves (columnas) del DataFrame
for key in df.keys():
    print(key)

# Crear una matriz de características combinando los ingresos anuales ("Annual Income (k$)") y
# las puntuaciones de gasto ("Spending Score (1-100)")
X = np.array(list(zip(f1,f2)))

# Mostrar los datos en 2D (Annual Income (k$) vs Spending Score (1-100))
plt.scatter(f1,f2, c='red',s=20)
#plt.show()


# Importar KMeans de scikit-learn para realizar clustering
from sklearn.cluster import KMeans


# Inicializar y ajustar el modelo KMeans con 3 clústeres
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)  # Predecir las etiquetas de clúster para cada punto de datos
C = kmeans.cluster_centers_ # Obtener los centros de los clústeres


# Visualización en 3D
fig = plt.figure() # Crear una nueva figura
ax = fig.add_subplot(111, projection='3d') # Agregar un subplot 3D a la figura

# Mostrar los puntos de datos en 3D, con colores basados en las etiquetas de clúster
ax.scatter(X[:,0],X[:,1],X[:,1], c='y')
ax.scatter(C[:,0],C[:,1],C[:,1], marker="*", c='#050505', s=1000)

# Añadir título y etiquetas a los ejes del gráfico 3D
ax.set_title("3D Clusters and their centroids")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_zlabel("Spending Score (1-100)")
plt.show()