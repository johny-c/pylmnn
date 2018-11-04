from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces

from pylmnn import LargeMarginNearestNeighbor


data = fetch_olivetti_faces()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

knn = KNeighborsClassifier(n_neighbors=3)

# Train with no transformation (euclidean metric)
knn.fit(X_train, y_train)

# Test with euclidean metric
acc = knn.score(X_test, y_test)

print('KNN accuracy on test set: {}%'.format(acc))


# LMNN is no longer a classifier but a transformer
lmnn = LargeMarginNearestNeighbor(n_neighbors=3, verbose=1)
lmnn.fit(X_train, y_train)

# Train with transformation learned by LMNN
knn.fit(lmnn.transform(X_train), y_train)

# Test with transformation learned by LMNN
acc = knn.score(lmnn.transform(X_test), y_test)

print('LMNN accuracy on test set: {}%'.format(acc))
