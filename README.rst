PyLMNN
======

**PyLMNN** is an implementation of the `Large Margin Nearest
Neighbor <#paper>`__ algorithm for metric learning in pure python.

This implementation follows closely the original MATLAB code by Kilian
Weinberger found at https://bitbucket.org/mlcircus/lmnn. This version
solves the unconstrained optimisation problem and finds a linear
transformation using L-BFGS as the backend optimizer.

This package also uses Bayesian Optimization to find the optimal
hyper-parameters for LMNN using the excellent
`GPyOpt <http://github.com/SheffieldML/GPyOpt>`__ package.

Installation
^^^^^^^^^^^^

The code was developed in python 3.5 under Ubuntu 16.04. You can clone
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
-  GPy>=1.5.6
-  GPyOpt>=1.0.3
-  matplotlib>=1.5.3

Usage
^^^^^

Here is a minimal use case:

.. code-block:: python

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
    plot_comparison(clf.L, x_te, y_te, dim_pref=3)


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
