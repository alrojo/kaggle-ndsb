# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:44:22 2015

@author: alexander
"""
from skimage import io
import glob
import numpy as np

paths_train = glob.glob("data/imgs/*")
shapes = np.zeros((len(paths_train), 2))

for k, path in enumerate(paths_train):
    img = io.imread(path, as_grey=True)
    print k
    shapes[k] = img.shape

np.save("shapes.npy", shapes)