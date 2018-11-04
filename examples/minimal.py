from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from pylmnn import LargeMarginNearestNeighbor as LMNN


# Load a data set
X, y = load_iris(return_X_y=True)

# Split in training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

# Set up the hyperparameters
k_train, k_test, n_components, max_iter = 3, 3, X.shape[1], 180

# Instantiate the metric learner
lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter, n_components=n_components)

# Train the metric learner
lmnn.fit(X_train, y_train)

# Fit the nearest neighbors classifier
knn = KNeighborsClassifier(n_neighbors=k_test)
knn.fit(lmnn.transform(X_train), y_train)

# Compute the k-nearest neighbor test accuracy after applying the learned transformation
lmnn_acc = knn.score(lmnn.transform(X_test), y_test)
print('LMNN accuracy on test set of {} points: {:.4f}'.format(X_test.shape[0], lmnn_acc))
