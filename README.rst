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

The simplest use case would be something like:

.. literalinclude:: ./examples/minimal.py
   :language: python
   :linenos:

You can check the examples directory for a demonstration of how to use the
code with different datasets and how to estimate good hyperparameters with Bayesian Optimisation.

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

This work is released under the `GNU General Public License Version 3
(GPLv3) <http://www.gnu.org/licenses/gpl.html>`__.

Contact **John Chiotellis**
`:envelope: <mailto:johnyc.code@gmail.com>`__ for questions, comments
and reporting bugs.
