# -*- encoding: utf-8 -*-

import sys
import argparse

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ml.classification import kNN
from mnist import loadmnist
import timestat


def init_parser():
    parser = argparse.ArgumentParser(description='Digit recognition home project')
    parser.add_argument('-x_train_dir', required=True, help='training set images')
    parser.add_argument('-y_train_dir', required=True, help='training set labels')
    parser.add_argument('-model_output_dir', default='./model.zip')
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


@timestat.timeit('Training')
def get_model(X, y):
    return kNN(k=3).fit(
        X, y, 
        max_features_per_class=[
             700, # 0
             700, # 1
            1500, # 2
            1900, # 3
            1900, # 4
            1500, # 5
            1000, # 6
            1500, # 7
            1500, # 8
            1500, # 9
        ]
    )


@timestat.timeit('Validate model')
def predict(model, X):
    return model(X)


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    X, y = load_dataset(args.x_train_dir, args.y_train_dir)
    X = prepare_dataset(X)

    model = get_model(X, y)
    with open(args.model_output_dir, 'wb') as fout:
        model.dump(fout)

    y_pred = predict(model, X)

    print(classification_report(y, y_pred), file=sys.stderr)
