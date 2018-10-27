# -*- encoding: utf-8 -*-

import sys
import argparse

import numpy as np
from sklearn.metrics import classification_report

from mnist import loadmnist
from ml.classification import kNN
import timestat


def init_parser():
    parser = argparse.ArgumentParser(description='Digit recognition home project (model checker)')
    parser.add_argument('-x_test_dir', required=True, help='test set images')
    parser.add_argument('-y_test_dir', required=True, help='test set labels')
    parser.add_argument('-model_input_dir', default='./model.zip')
    return parser


@timestat.timeit('Reading dataset')
def load_dataset(x_file_name, y_file_name):
    X, y = loadmnist(x_file_name, y_file_name)
    return X, y


@timestat.timeit('Preparing dataset')
def prepare_dataset(X):
    n = 28
    d = 5
    s = 3
    res = []
    for x in X:
        x = x.reshape(n, n)[d:n-d,d:n-d]
        _x = []
        for i in range(0, n - 2*d, s):
            for j in range(0, n - 2*d, s):
                _x.append(int(np.sum(x[i:i+s,j:j+s]) / s**2))
        res.append(_x)
    X = np.array(res, dtype=np.int16)
    return X


@timestat.timeit('Reading model')
def get_model(model_file):
    with open(model_file, 'rb') as fin:
        model = kNN().load(fin)
    return model


@timestat.timeit('Validate model')
def predict(model, X):
    return model(X)


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    X, y = load_dataset(args.x_test_dir, args.y_test_dir)
    X = prepare_dataset(X)
    model = get_model(args.model_input_dir)
    
    y_pred = predict(model, X)
    print(classification_report(y, y_pred), file=sys.stderr)
