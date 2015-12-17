import sys
import numpy as np
import importlib
import lasagne as nn
import theano
from theano import tensor as T
import os
import glob

import data
import utils

sym_y = T.imatrix('target_output')
sym_x = T.tensor4()

metadata_path_all = glob.glob(sys.argv[1])

print "shape of metadata_path_all"
print(len(metadata_path_all))
# metadata
print "Loading metadata file %s" % metadata_path_all
metadata = np.load(metadata_path_all)
config_name = metadata['config_name']
config = importlib.import_module("configurations.%s" % config_name)
print "Using configurations: '%s'" % config_name

# Image_gen
print "Building image_gen ..."
image_gen = data.gen_images(data.paths['test'], labels=None, shuffle=False, repeat=False, name="eval_train_gen")    
config.data_loader.load_test(image_gen)
gen = config.data_loader.create_fixed_gen(config.data_loader.image_gen_test, augment=augment)

# Model
print "Building model ..."
l_in, l_out = config.build_model()
print "Build eval function"

inference = nn.layers.get_output(
    l_out, sym_x, deterministic=True)

print "Load parameters"
nn.layers.set_all_param_values(l_out, metadata['param_values'])

print "Compile functions"
predict = theano.function([sym_x], inference)

print "Predict"

predictions = []
batch_size = 32

idx = range(0, batch_size)
x_batch = next(gen)
p = predict(x_batch)
predictions_path = "preds.npy"
print "Storing predictions in %s" % predictions_path
np.save(predictions_path, p)