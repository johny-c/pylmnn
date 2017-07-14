from sklearn.utils.estimator_checks import check_estimator
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN

check_estimator(LMNN)  # passes
