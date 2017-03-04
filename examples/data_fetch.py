import numpy as np
import csv, os
from sklearn.datasets import get_data_home
from sklearn import datasets as skd
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split


def fetch_from_config(cfg):
    data_set_name = cfg['fetch']['name']
    if cfg['fetch'].getboolean('sklearn'):
        if data_set_name == 'OLIVETTI':
            data_set = skd.fetch_olivetti_faces(shuffle=True)
        else:
            data_set = skd.fetch_mldata(data_set_name)
        X, y = data_set.data, data_set.target
        if data_set_name == 'MNIST original':
            if cfg['pre_process'].getboolean('normalize'):
                X = X / 255.
    else:
        if data_set_name == 'LETTERS':
            X, y = fetch_load_letters()
        elif data_set_name == 'ISOLET':
            x_tr, x_te, y_tr, y_te = fetch_load_isolet()
        elif data_set_name == 'SHREC14':
            X, y = load_shrec14(real=cfg['fetch']['real'], desc=cfg['fetch']['desc'])
            X = prep.normalize(X, norm=cfg['pre_process']['norm'])
        else:
            raise NameError('No data set {} found!'.format(data_set_name))

    # Separate training and testing set
    if data_set_name == 'MNIST original':
        x_tr, x_te, y_tr, y_te = X[:60000], X[60000:], y[:60000], y[60000:]
    elif data_set_name != 'ISOLET':
        test_size = cfg['train_test'].getfloat('test_size')
        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y)

    return x_tr, x_te, y_tr, y_te


def fetch_load_letters(data_dir=None):
    path = os.path.join(get_data_home(data_dir), 'letter-recognition.data')

    if not os.path.exists(path):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        print('Downloading letter-recognition dataset from {}...'.format(url))
        request.urlretrieve(url=url, filename=path)
    else:
        print('Found letter-recognition in {}!'.format(path))

    X, y = [], []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            y.append(row[0])
            X.append(row[1:])
    labels, label_idx = np.unique(y, return_inverse=True)
    return np.asarray(X, dtype=float), label_idx


def decompress_z(fname_in, fname_out=None):
    from unlzw import unlzw
    fname_out = fname_in[:-2] if fname_out is None else fname_out
    print('Extracting {} to {}...'.format(fname_in, fname_out))
    with open(fname_in, 'rb') as fin, open(fname_out, 'wb') as fout:
        compressed_data = fin.read()
        uncompressed_data = unlzw(compressed_data)
        fout.write(uncompressed_data)


def fetch_load_isolet(data_dir=None):
    train = 'isolet1+2+3+4.data.Z'
    test = 'isolet5.data.Z'
    path_train = os.path.join(get_data_home(data_dir), train)
    path_test  = os.path.join(get_data_home(data_dir), test)

    if not os.path.exists(path_train[:-2]) or not os.path.exists(path_test[:-2]):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/'
        if not os.path.exists(path_train[:-2]):
            if not os.path.exists(path_train):
                print('Downloading Isolated Letter Speech Recognition data set from {}...'.format(
                    url))
                request.urlretrieve(url=url+train, filename=path_train)
            # os.system('gzip -d ' + path_train)
            decompress_z(path_train)
        if not os.path.exists(path_test[:-2]):
            if not os.path.exists(path_test):
                print('Downloading Isolated Letter Speech Recognition data set from {}...'.format(
                    url))
                request.urlretrieve(url=url+test, filename=path_test)
            # os.system('gzip -d ' + path_test)
            decompress_z(path_test)
    else:
        print('Found Isolated Letter Speech Recognition data set!')

    xtr, ytr = [], []
    with open(path_train[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            xtr.append(row[:-1])
            ytr.append(int(float(row[-1])))
    labels, ytr = np.unique(ytr, return_inverse=True)
    xte, yte = [], []
    with open(path_test[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            xte.append(row[:-1])
            yte.append(int(float(row[-1])))
    labels, yte = np.unique(yte, return_inverse=True)

    return np.asarray(xtr, dtype=float), np.asarray(xte, dtype=float), ytr, yte


def load_shrec14(data_dir='shrec14_data', real=False, desc='csd'):
    from os.path import join
    import scipy.io as sio
    f = 'desc_shrec14_real.mat' if real else 'desc_shrec14_synth.mat'
    mat_dict = sio.loadmat(join(data_dir, f))

    if desc == 'sihks':
        data = mat_dict['SihksPooledMat']
    elif desc == 'wks':
        data = mat_dict['WksPooledMat']
    elif desc == 'csd':
        data = np.hstack((mat_dict['WksPooledMat'], mat_dict['SihksPooledMat']))
    else:
        raise TypeError('No descriptor named %s is known.' % desc)

    labels = mat_dict['C']
    # Make labels start from 0 instead of 1
    labels -= 1 if min(labels) == 1 else None
    labels = np.array(labels.T[0]).T

    return data, labels


