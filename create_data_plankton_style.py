import os

import numpy as np
import pandas as pd
import skimage.io as sk



def rgb2gray(rbg):
    return np.dot(rbg[...,:3], [0.299, 0.587, 0.144])

csv = pd.csv_read('./trans.csv')
whaleIDs = list(set(csv['whaleID']))

print "Making directories ..."
for i in range(len(whaleIDs)):
    path = "%s%s%s" (os.getcwd(), "/data/", whaleIDs[i])
    os.mkdir(path, 0777)

print "Filling directories ..."
for i in range(len(csv['Image'])):
    path_from = "%s%s%s" (os.getcwd(), "/data/orig/train/", (csv['image'][i]))
    path_to = "%s%s%s%s" (os.getcwd(), "/data/train/", (csv['whaleID'][i]), (csv['image'][i]))
    image = sk.imread(path_from)
    sk.imsave(path)