from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from pylmnn import find_hyperparams


def test_find_hyperparams():
    X, y = load_iris(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    best_params = find_hyperparams(X_train, y_train, X_valid, y_valid, params={}, max_bopt_iter=5)

