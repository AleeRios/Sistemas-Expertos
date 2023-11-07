import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

db_iris = load_iris()
#print(type(db_iris))

#print(db_iris.keys())

print(db_iris['data'])

print(db_iris['target_names'])

print(db_iris['target'])

print(db_iris['feature_names'])

x_entrenar, x_test, y_entrenar, y_test = train_test_split(db_iris['data'], db_iris['target'])

print(x_entrenar.shape)

print(y_entrenar.shape)

##clasificador
knn = KNeighborsClassifier( n_neighbors=10)
knn.fit(x_entrenar, y_entrenar)

aprend = knn.score(x_test, y_test)
print('{:.5f}'.format(aprend))

#Virginica
clasif = knn.predict([[8.0, 4.0, 6.4, 2.0]])
print(clasif, ' = ', db_iris['target_names'][clasif])
clasif = knn.predict([[7.9, 3.8, 6.0, 2.12]])
print(clasif, ' = ', db_iris['target_names'][clasif])
clasif = knn.predict([[7.3, 3.3, 5.93, 1.9]])
print(clasif, ' = ', db_iris['target_names'][clasif])

#Versicolor
clasif = knn.predict([[6.9, 2.8, 5.0, 1.12]])
print(clasif, ' = ', db_iris['target_names'][clasif])
clasif = knn.predict([[6.5, 2.5, 5.2, 1.]])
print(clasif, ' = ', db_iris['target_names'][clasif])

#Setosa
clasif = knn.predict([[3.5, 3.8, 3.0, .9]])
print(clasif, ' = ', db_iris['target_names'][clasif])
clasif = knn.predict([[3.2, 3.1, 2.7, .3]])
print(clasif, ' = ', db_iris['target_names'][clasif])