import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Establecer el estilo de los gráficos de Seaborn
sns.set_style('white')

# Definir los nombres de las columnas y cargar los datos desde un archivo de texto
colum_names = ['user_id', 'item_id' ,'rating' ,'timestamp']
df = pd.read_csv('../u.data', sep='\t', names=colum_names)
# print(df.head())

# Cargar los títulos de las películas desde otro archivo
titulos_peliculas = pd.read_csv("../Movie_Id_Titles")
# print(titulos_peliculas.head())

# Fusionar los datos de puntuacion (ratings) con los títulos de las películas utilizando 'item_id' como clave
df = pd.merge(df, titulos_peliculas, on='item_id')
# print(df.head())
# print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())
# print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

# Crear un DataFrame que contiene la media de las puntuaciones por título de película
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
# print(ratings.head())

# Añadir una columna con el número de puntuaciones por título
ratings['numero de puntuaciones'] = pd.DataFrame(df.groupby('title')['rating'].count())
# print(ratings.head())

# Graficar el histograma del número de puntuaciones por película
plt.figure(figsize=(10,4))
ratings['numero de puntuaciones'].hist(bins=70)
# plt.show()

# Graficar el histograma de las puntuaciones medias por película
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
#plt.show()

# Graficar una relación conjunta entre la media de las puntuaciones y el número de puntuaciones
sns.jointplot(x='rating', y='numero de puntuaciones', data = ratings,  alpha=0.5)
#plt.show()

# Crear una tabla dinámica de usuarios y títulos de películas con las puntuaciones
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
#print(moviemat.head())
#print(ratings.sort_values('numero de puntuaciones', ascending=False).head(10))

# Seleccionar las puntuaciones de dos películas específicas: 'Star Wars (1977)' y 'Liar Liar (1997)'
puntuaciones_starwars = moviemat['Star Wars (1977)']
puntuaciones_liarliar = moviemat['Liar Liar (1997)']
# print(puntuaciones_starwars.head())
# print(puntuaciones_liarliar.head())

relacion_starwars = moviemat.corrwith(puntuaciones_starwars)
relacion_liarliar = moviemat.corrwith(puntuaciones_liarliar)
#print(relacion_starwars)
#print(relacion_liarliar)
# Si compruebas esto, dara errores, puesto que hay valores nulos dentro de la tabla,
# lo que hace que la tarea no se realice como es debido

# Calcular la correlación de todas las películas con 'Star Wars (1977)'
correlacion_starwars = pd.DataFrame(relacion_starwars, columns=['Correlacion'])
correlacion_starwars.dropna(inplace=True)
#print(correlacion_starwars.head())
#print(correlacion_starwars.sort_values('Correlacion', ascending=False).head(10))

# Añadir el número de puntuaciones al DataFrame de correlación de 'Star Wars (1977)'
correlacion_starwars = correlacion_starwars.join(ratings['numero de puntuaciones'])
#print(correlacion_starwars.head())

# Filtrar y mostrar las 10 películas más correlacionadas con 'Star Wars (1977)' que tienen más de 100 puntuaciones
print(correlacion_starwars[correlacion_starwars['numero de puntuaciones']>100].sort_values('Correlacion', ascending=False).head())

# Repetir el proceso para 'Liar Liar (1997)'
correlacion_liarliar = pd.DataFrame(relacion_liarliar, columns=['Correlacion'])
correlacion_liarliar.dropna(inplace=True)
correlacion_liarliar = correlacion_liarliar.join(ratings['numero de puntuaciones'])

# Filtrar y mostrar las 5 películas más correlacionadas con 'Liar Liar (1997)' que tienen más de 100 puntuaciones
print(correlacion_liarliar[correlacion_liarliar['numero de puntuaciones']>100].sort_values('Correlacion', ascending=False).head())