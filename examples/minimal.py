from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
from pylmnn.plots import plot_comparison


# Load a data set
dataset = load_iris()
X, y = dataset.data, dataset.target

# Split in training and testing set
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

# Set up the hyperparameters
k_tr, k_te, dim_out, max_iter = 3, 1, X.shape[1], 180

# Instantiate the classifier
clf = LMNN(n_neighbors=k_tr, max_iter=max_iter, n_features_out=dim_out)

# Train the classifier
clf = clf.fit(x_tr, y_tr)

# Compute the k-nearest neighbor test accuracy after applying the learned transformation
accuracy_lmnn = clf.score(x_te, y_te)
print('LMNN accuracy on test set of {} points: {:.4f}'.format(x_te.shape[0], accuracy_lmnn))

# Draw a comparison plot of the test data before and after applying the learned transformation
plot_comparison(clf.L_, x_te, y_te, dim_pref=3)
