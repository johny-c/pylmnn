from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
from pylmnn.helpers import test_knn, plot_ba


# Load a data set
dataset = fetch_mldata('iris')
X, y = dataset.data, dataset.target

# Split in training and testing set
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

# Set up the hyperparameters
k_tr, k_te, dim_out, max_iter = 3, 1, X.shape[1], 180

# Instantiate the classifier
clf = LMNN(k=k_tr, max_iter=max_iter, dim_out=dim_out)

# Train the classifier
clf = clf.fit(x_tr, y_tr)

# Compute the k-nearest neighbor test accuracy after applying the learned transformation
test_acc = test_knn(x_tr, y_tr, x_te, y_te, k=k_te, L=clf.L)

# Draw a comparison plot of the test data before and after applying the learned transformation
plot_ba(clf.L, x_te, y_te, dim_pref=3)
