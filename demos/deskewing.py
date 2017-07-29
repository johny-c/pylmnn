import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import interpolation
from sklearn.datasets import get_data_home

# Deskewing code taken from https://fsix.github.io/mnist/Deskewing.html


def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


def main():
    mnist = loadmat(os.path.join(get_data_home(), 'mldata',
                                 'mnist-original.mat'))

    X = mnist['data'].T
    X = np.asarray(X, dtype=np.float64)
    X2 = np.zeros_like(X)

    for i in range(len(X)):
        X2[i] = deskew(X[i].reshape(28, 28)).ravel()

    return X2

if __name__ == '__main__':
    main()