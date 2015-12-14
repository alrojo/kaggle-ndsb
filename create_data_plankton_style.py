import os
import glob
import shutil

import numpy as np
import pandas as pd
import skimage.io as sk
from scipy import ndimage
from scipy import misc

csv = pd.read_csv('./data/train.csv')
whaleIDs = list(set(csv['whaleID']))
train_paths = csv['Image']
paths = glob.glob('./data/imgs/*')
base_paths = [os.path.basename(p) for p in paths]
test_paths = list(set(base_paths).difference(set(train_paths)))

print "total path size: %d" % len(paths)
print "size of train_paths %d" % len(train_paths)
print "size of test_paths %d" % len(test_paths)
print "size haleIDs %d" % len(whaleIDs)

assert len(paths) == (len(train_paths) + len(test_paths))
assert len(whaleIDs) == 447

#print "Making directories ..."
#for i in range(len(whaleIDs)):
#    path = "%s%s%s" % (os.getcwd(), "/data/train/", whaleIDs[i])
#    print path
#    os.mkdir(path, 0777)

print "Filling directories ..."
for i in range(len(train_paths)):
    path_from = "%s%s%s" % (os.getcwd(), "/data/imgs/", (train_paths[i]))
    path_to = "%s%s%s%s%s" % (os.getcwd(), "/data/train/", (csv['whaleID'][i]), "/", (train_paths[i]))
    img = ndimage.imread(path_from)
    rescaled = ndimage.zoom(img, 0.25)
#    shutil.move(path_from, path_to)

for path in test_paths:
    path_from = "%s%s%s" % (os.getcwd(), "/data/imgs/", path)
    path_to = "%s%s%s" % (os.getcwd(), "/data/test/", path)
    img = ndimage.imread(path_from)
    rescaled = ndimage.zoom(img, 0.25)
    misc.imsave(path_to, rescaled)
#    shutil.move(path_from, path_to)
