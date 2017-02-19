# pylmnn

**pylmnn** is a python implementation of the [Large Margin Nearest Neighbor](#paper)  algorithm for metric learning.
 
 This implementation follows closely the original MATLAB code by Kilian Weinberger found at [https://bitbucket.org/mlcircus/lmnn](https://bitbucket.org/mlcircus/lmnn). This version solves the unconstrained optimisation problem and finds a linear transformation using L-BFGS as the backend optimizer.

 This package also uses Bayesian Optimization to find the optimal hyper-parameters for LMNN using the excellent [GPyOpt](http://github.com/SheffieldML/GPyOpt) package.


#### Installation

The code was developed in python 3.5 under Ubuntu 16.04. You can clone the repo with:
```
git clone https://github.com/johny-c/pylmnn.git
```

or install it via pip:
```
pip3 install pylmnn
```

#### Dependencies

* numpy>=1.11.2
* scipy>=0.18.1
* scikit_learn>=0.18.1
* GPyOpt>=1.0.3
* matplotlib>=1.5.3

#### Usage

The simplest use case would be something like:

    X, y = load_my_data(dataset_name)
    xtr, xte, ytr, yte = train_test_split(X, y, train_size=0.5, stratify=y)
    Klmnn, Knn, outdim, maxiter = 3, 1, X.shape[1], 180
    lmnn = LMNN(verbose=True, k=Klmnn, max_iter=maxiter, outdim=outdim)
    lmnn, loss, details = lmnn.fit(xtr, ytr)
    test_acc = test_knn(xtr, ytr, xte, yte, k=Knn, L=lmnn.L)

You can check the examples directory for examples of how to use the code.

#### Disclaimer

This is still under development. The current version of the code is fast but can be very memory hungry.

#### References

If you use this code in your work, please cite the following <a name="paper">publication</a>.

    @ARTICLE{weinberger09distance,
        title={Distance metric learning for large margin nearest neighbor classification},
        author={Weinberger, K.Q. and Saul, L.K.},
        journal={The Journal of Machine Learning Research},
        volume={10},
        pages={207--244},
        year={2009},
        publisher={MIT Press}
    }


#### License and Contact

This work is released under the [GNU General Public License Version 3 (GPLv3)](http://www.gnu.org/licenses/gpl.html).

Contact **John Chiotellis** [:envelope:](mailto:johnyc.code@gmail.com) for questions, comments and reporting bugs.
