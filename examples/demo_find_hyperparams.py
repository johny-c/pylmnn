from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from pylmnn.bayesopt import find_hyperparams


X, y = load_iris(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
best_params = find_hyperparams(X_train, y_train, X_valid, y_valid, params={}, max_bopt_iter=5)
best_n_neighbors, best_n_neighbors_predict, best_n_components, best_max_iter = best_params
