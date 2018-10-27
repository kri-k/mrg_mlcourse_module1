# -*- encoding: utf-8 -*-

import struct

import numpy as np


def loadmnist(imagefile, labelfile):
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = struct.unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = struct.unpack('>I', rows)[0]
    cols = images.read(4)
    cols = struct.unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = struct.unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = struct.unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = struct.unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)