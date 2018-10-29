# -*- encoding: utf-8 -*-

import struct

import numpy as np


def loadmnist(imagefile, labelfile):
    with open(imagefile, 'rb') as images:
        images.read(4)  # skip the magic_number
        number_of_images = struct.unpack('>I', images.read(4))[0]
        rows = struct.unpack('>I', images.read(4))[0]
        cols = struct.unpack('>I', images.read(4))[0]
        x = np.frombuffer(images.read(), dtype=np.uint8).reshape((number_of_images, rows, cols))

    with open(labelfile, 'rb') as labels:
        labels.read(4) # skip the magic_number
        N = struct.unpack('>I', labels.read(4))[0]
        y = np.frombuffer(labels.read(), dtype=np.uint8)

    return (x, y)
