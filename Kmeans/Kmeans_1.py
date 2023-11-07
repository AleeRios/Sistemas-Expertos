import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn import metrics

db_iris = load_iris()

datos = db_iris.data            #datos BD
clases = db_iris.target         #clases -> etiquetas    #print(db_iris['target'])

print(datos)
#print(type(clases))

#Clustering
kmeans = KMeans( n_clusters=3 )     #n_cluster = 8 default   init = k-means++
#kmeans = KMeans(n_clusters = 3, init='random')

kmeans.fit( datos )             #Clusters

print( kmeans.labels_ )         #Clases

print( kmeans.cluster_centers_ )    #Centroides (últimos)
    
print( kmeans.n_iter_ )


predic = kmeans.predict(datos)      #Clases  == labels_
print( predic )

predic2 = kmeans.fit_predict( datos )       #fit + predict
print( predic2 )

score = metrics.adjusted_rand_score( clases, predic)
print( score)



#plt.figure('K-Means')     
#plt.subplot(211)    #FCN
#plt.scatter(datos[:,0], datos[:,3], c=predic)   #X, Y ... C=color

#plt.subplot(212)
#plt.scatter(datos[:,0], datos[:,3], c=clases)

fig = plt.figure('K-Means')
ax = fig.add_subplot(121, projection='3d')
ax.scatter3D(datos[:,0], datos[:,2], datos[:,3], c=predic2)
ax.set_xlabel("X")      #Sepal      l
ax.set_ylabel("Y")      #Sepal      a
ax.set_zlabel("Z")      #Petalo     l

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter3D(datos[:,0], datos[:,2], datos[:,3], c=clases)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.show()
