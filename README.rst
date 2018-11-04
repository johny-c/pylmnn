PyLMNN
======

**PyLMNN** is an implementation of the `Large Margin Nearest
Neighbor <#paper>`__ algorithm for metric learning in pure python.

This implementation follows closely the original MATLAB code by Kilian
Weinberger found at https://bitbucket.org/mlcircus/lmnn. This version
solves the unconstrained optimisation problem and finds a linear
transformation using L-BFGS as the backend optimizer.

This package can also  find optimal
hyper-parameters for LMNN via Bayesian Optimization using the excellent
`GPyOpt <http://github.com/SheffieldML/GPyOpt>`__ package.

Installation
^^^^^^^^^^^^

The code was developed in python 3.5 under Ubuntu 16.04 and was also tested under Ubuntu 18.04 and python 3.6. You can clone
the repo with:

::

    git clone https://github.com/johny-c/pylmnn.git

or install it via pip:

::

    pip3 install pylmnn

Dependencies
^^^^^^^^^^^^

-  numpy>=1.11.2
-  scipy>=0.18.1
-  scikit\_learn>=0.18.1

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

In case you want to use the hyperparameter optimization module, you should also install:

-  GPy>=1.5.6
-  GPyOpt>=1.0.3

Usage
^^^^^

Here is a minimal use case:

.. code-block:: python

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

You can check the examples directory for a demonstration of how to use the
code with different datasets and how to estimate good hyperparameters with Bayesian Optimisation.

Documentation can also be found at http://pylmnn.readthedocs.io/en/latest/ .

References
^^^^^^^^^^

If you use this code in your work, please cite the following
publication.

::

    @ARTICLE{weinberger09distance,
        title={Distance metric learning for large margin nearest neighbor classification},
        author={Weinberger, K.Q. and Saul, L.K.},
        journal={The Journal of Machine Learning Research},
        volume={10},
        pages={207--244},
        year={2009},
        publisher={MIT Press}
    }

License and Contact
^^^^^^^^^^^^^^^^^^^

This work is released under the `3-Clause BSD License <https://opensource.org/licenses/BSD-3-Clause>`__.

Contact **John Chiotellis**
`:envelope: <mailto:johnyc.code@gmail.com>`__ for questions, comments
and reporting bugs.
