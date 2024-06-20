import pandas as pd
# Clasificador del arbol de decision
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics

# Cargamos el dataset, con columnas personalizadas
col_names = ['company', 'job','degree', 'salary_more_than_100k']
df = pd.read_csv("../salaries.csv", header=None, names=col_names)

# Descomentar para saber ver las primeras filas del dataset
# print(df)

from sklearn import preprocessing

# Preprocesamiento: codificamos las variables categoricas a numericas
label_encoder = preprocessing.LabelEncoder()
df['company']=label_encoder.fit_transform(df['company'])
df['job']=label_encoder.fit_transform(df['job'])
df['degree']=label_encoder.fit_transform(df['degree'])
# Descomentar para ver el dataset transformado
#print(df.head())

# Definimos las caracteristicas y las variables objetivo
feature_cols = ['company', 'job', 'degree']
X = df[feature_cols]
Y = df['salary_more_than_100k']
#X = df.values[1:,:3]
#Y = df.values[1:,3]
#print(X)
#print(Y)

# Dividimos los datos en conjunto de entrenamiento y de prueba
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size=0.2,random_state=100)

# Creamos un objeto clasificador de arbol de decision usando entropia
# 1. El primer parametro es el criterio que vamos a usar para entrenar al arbol de decision, en nuestro caso, la entropia
# 2. El segundo parametro es la profundida que tendra nuestro arbol de decision, en nuestro caso, 2
clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=8)

# Entrenamos el clasificador del arbol de decision

clf_entropy = clf_entropy.fit(X_train,Y_train)

# Predecir la repsuesta para el conjunto de datos de prueba

Y_pred = clf_entropy.predict(X_test)

print("Accuracy", metrics.accuracy_score(Y_test,Y_pred))

