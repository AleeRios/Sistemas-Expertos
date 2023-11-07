
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_svmlight_file

def clasificar(clasif):
    if clasif == 0:
        print(clasif, "= setosa")
    elif clasif == 1:
        print(clasif, "= Versicolor")
    elif clasif == 2:
        print(clasif, "Virginica")
    else:
        print(clasif, "Orquidea")

archivo = 'Iris_DB_svm.txt'
x_train, y_train = load_svmlight_file(archivo)

print(x_train)
print(y_train)

print( type(x_train) )
print( type(y_train) )


x_entrenar, x_test, y_entrenar, y_test = train_test_split(x_train, y_train)



print(x_entrenar.shape)

print(y_entrenar.shape)

##clasificador
knn = KNeighborsClassifier( n_neighbors=10)

knn.fit(x_entrenar, y_entrenar)

aprend = knn.score(x_test, y_test)
print('{:.5f}'.format(aprend))

#2 nuevas clasificaciones (Orquidea)
clasif = knn.predict([[8.2, 3.7, 6.8, 3.1]])
clasificar(clasif)

clasif = knn.predict([[8.1, 3.9, 7.1, 3.6]])
clasificar(clasif)

#2 antiguas Clasificaciones (Versicolor || virginica || setosa)
clasif = knn.predict([[1.2, 3.4, 5.6, 1.1]])
clasificar(clasif)

clasif = knn.predict([[3.2, 4.4, 6.6, 2.1]])
clasificar(clasif)
