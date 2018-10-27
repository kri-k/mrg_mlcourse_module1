# -*- encoding: utf-8 -*-

from collections import defaultdict
from collections import Counter
import heapq
import io
import pickle
import random
import zipfile

import numpy as np


class kNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None
        self.ord = 2

    def fit(self, X, y, max_features_per_class=1000, norm_ord=2):
        try:
            iter(max_features_per_class)
            max_features_per_class = tuple(max_features_per_class)
        except TypeError:
            max_features_per_class = defaultdict(lambda k=max_features_per_class: k)
        self.X = []
        self.y = []
        self.ord = norm_ord
        for label in set(y):
            s = random.sample(tuple(X[y == label]), max_features_per_class[int(label)])
            self.X.extend(s)
            self.y.extend((label,) * len(s))
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        return self

    def _find_nearest_neighbor(self, target_x):
        dists = np.linalg.norm(self.X - target_x, ord=self.ord, axis=1)
        return Counter(self.y[np.argpartition(dists, self.k)[:self.k]]).most_common(1)[0][0]

    def _predict(self, X):
        return np.array(tuple(self._find_nearest_neighbor(x) for x in X))

    def __call__(self, X):
        return self._predict(X)

    def dump(self, fout):
        in_mmr_zip = io.BytesIO()
        with zipfile.ZipFile(in_mmr_zip, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr('x', pickle.dumps(self.X))
            zip_file.writestr('y', pickle.dumps(self.y))
            zip_file.writestr('ord', pickle.dumps(self.ord))
            zip_file.writestr('k', pickle.dumps(self.k))
        fout.write(in_mmr_zip.getvalue())

    def load(self, fin):
        with zipfile.ZipFile(fin, 'r') as zip_file:
            self.X = pickle.loads(zip_file.read('x'))
            self.y = pickle.loads(zip_file.read('y'))
            self.k = int(pickle.loads(zip_file.read('k')))
            self.ord = pickle.loads(zip_file.read('ord'))
        try:
            self.ord = int(self.ord)
        except ValueError:
            pass
        return self


if __name__ == '__main__':
    pass
