import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize']=(16,9)

X,Y = make_blobs(n_samples=800, n_features=3, centers=4)

from sklearn.cluster import KMeans

wcss_list = []

for i in range (1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state=42)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)


plt.plot(range(1,11),wcss_list)
plt.title('El metodo Elbow')
plt.xlabel('Numero de clusteres')
plt.ylabel('wcss_list')
plt.show()