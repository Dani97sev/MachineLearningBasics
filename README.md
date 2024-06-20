# MachineLearningBasics

Recursos y ejemplos en aprendizaje automático. Incluye tutoriales sobre modelos, algoritmos y herramientas comunes.

## Índice
1. [Introducción](#introducción)
2. [Descripción de Archivos](#descripción-de-archivos)
   - [1. ArbolDeDecision.py](#1-arboldedecisionpy)
   - [2. Ejemplo_Clustering_KMeans.py](#2-ejemplo_clustering_kmeanspy)
   - [3. KMeans.py](#3-kmeanspy)
   - [4. KNN.py](#4-knnpy)
   - [5. PrediccionPreciosViviendaRLS.py](#5-prediccionpreciosviviendarlspy)
   - [6. RegresionLogistica.py](#6-regresionlogisticapy)
   - [7. RLM.py](#7-rlmpy)
   - [8. SistemaDeRecomendacion.py](#8-sistemaderecomendacionpy)

## Introducción

El proyecto "Curso de Machine Learning" contiene una serie de scripts en Python diseñados para ilustrar y enseñar diferentes algoritmos y técnicas de aprendizaje automático. Cada archivo aborda un aspecto específico del Machine Learning, desde modelos de regresión y clasificación hasta sistemas de recomendación y clustering. A continuación se presenta una descripción detallada de cada archivo, sus funciones principales y su propósito dentro del curso.

## Descripción de Archivos

### 1. ArbolDeDecision.py
**Descripción**: Este script implementa un modelo de árbol de decisión para resolver problemas de clasificación o regresión.

**Funciones Principales**:
- **Cargar y procesar datos**: Lectura de conjuntos de datos y preparación para el entrenamiento del modelo.
- **Entrenar el modelo**: Uso de un árbol de decisión para aprender patrones a partir de los datos.
- **Evaluar el modelo**: Medición del rendimiento del modelo utilizando métricas como exactitud, precisión y recall.

### 2. Ejemplo_Clustering_KMeans.py
**Descripción**: Este script proporciona un ejemplo práctico de cómo usar el algoritmo de clustering K-Means para agrupar datos no etiquetados.

**Funciones Principales**:
- **Cargar y procesar datos**: Preparación de los datos para el clustering.
- **Entrenar el modelo**: Aplicación del algoritmo K-Means para identificar clusters en los datos.
- **Visualización**: Representación gráfica de los resultados del clustering.

### 3. KMeans.py
**Descripción**: Este archivo contiene una implementación del algoritmo K-Means desde cero, sin utilizar bibliotecas externas.

**Funciones Principales**:
- **Inicialización de centroides**: Selección inicial de puntos centrales para los clusters.
- **Asignación de puntos a clusters**: Clasificación de cada punto de datos según el centroide más cercano.
- **Recalibración de centroides**: Actualización de los centroides basándose en la media de los puntos asignados a cada cluster.
- **Iteración hasta la convergencia**: Repetición del proceso hasta que los centroides no cambien significativamente.

### 4. KNN.py
**Descripción**: Este script implementa el algoritmo K-Nearest Neighbors (KNN) para clasificación y regresión.

**Funciones Principales**:
- **Cargar y procesar datos**: Lectura y preparación de los datos de entrada.
- **Clasificación con KNN**: Predicción de la clase de nuevos datos basándose en la proximidad a puntos de datos conocidos.
- **Evaluación del modelo**: Uso de métricas de rendimiento para evaluar la precisión del modelo.

### 5. PrediccionPreciosViviendaRLS.py
**Descripción**: Este archivo predice precios de viviendas utilizando un modelo de Regresión Lineal Simple (RLS).

**Funciones Principales**:
- **Cargar datos de precios de viviendas**: Lectura de datos históricos de precios.
- **Entrenar el modelo**: Ajuste del modelo de regresión lineal a los datos.
- **Predicción y evaluación**: Predicción de precios de viviendas y evaluación de la precisión del modelo.

### 6. RegresionLogistica.py
**Descripción**: Este script implementa la regresión logística para resolver problemas de clasificación binaria.

**Funciones Principales**:
- **Cargar y procesar datos**: Preparación de datos para el entrenamiento del modelo.
- **Entrenar el modelo**: Ajuste del modelo de regresión logística a los datos de entrenamiento.
- **Evaluar el modelo**: Medición del rendimiento utilizando métricas como la precisión, recall y la curva ROC.

### 7. RLM.py
**Descripción**: Este archivo implementa la Regresión Lineal Múltiple (RLM) para predecir una variable dependiente utilizando múltiples variables independientes.

**Funciones Principales**:
- **Cargar y procesar datos**: Lectura y preparación de datos multivariantes.
- **Entrenar el modelo**: Ajuste del modelo de regresión lineal múltiple.
- **Análisis de coeficientes**: Interpretación de los coeficientes del modelo y su impacto en la variable dependiente.

### 8. SistemaDeRecomendacion.py
**Descripción**: Este script desarrolla un sistema de recomendación utilizando técnicas de filtrado colaborativo y basado en contenido.

**Funciones Principales**:
- **Cargar datos de usuarios y productos**: Lectura de datos sobre preferencias y comportamientos de los usuarios.
- **Generación de recomendaciones**: Uso de algoritmos para predecir productos de interés para los usuarios.
- **Evaluación del sistema**: Medición de la precisión y efectividad de las recomendaciones generadas.

