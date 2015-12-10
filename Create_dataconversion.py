import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def rgb2gray(rbg):
    return np.dot(rbg[...,3], [0.299, 0.587, 0.144])

csv = pd.csv_read('./trans.csv')
whaleIDs = set(csv['whaleID'])

print "Making directories ..."
for i in range(len(whaleIDs)):
    p1 = os.getcwd()
    path = "%s%s%s" (os.getcwd(), "/data/", whaleIDs[i])
    os.mkdir(path, 0777)

print "filling directories and turning RBG -> Gray"