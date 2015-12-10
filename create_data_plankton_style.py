import os
import glob

import numpy as np
import pandas as pd
import skimage.io as sk

csv = pd.csv_read('./data/train.csv')
whaleIDs = list(set(csv['whaleID']))
train_paths = csv['Image']
paths = glob.glob('./data/imgs/*')
base_paths = [os.path.basename(p) for p in paths]
test_paths = list(set(base_paths).difference(set(train_paths)))

print "total path size: %s" % paths.shape
print "size of train_paths %s" % train_paths.shape
print "size of test_paths %s" % test_paths.shape

assert False

print "Making directories ..."
for i in range(len(whaleIDs)):
    path = "%s%s%s" (os.getcwd(), "/data/", whaleIDs[i])
    os.mkdir(path, 0777)

print "Filling directories ..."
for i in range(len(train_paths)):
    path_from = "%s%s%s" (os.getcwd(), "/data/imgs/", (train_paths[i]))
    path_to = "%s%s%s%s" (os.getcwd(), "/data/train/", (csv['whaleID'][i]), (train_paths[i]))
    image = sk.imread(path_from)
    sk.imsave(path)


for path in test_paths:
    asd