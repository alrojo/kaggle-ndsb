import numpy as np

import theano 
import theano.tensor as T

import lasagne as nn
import lasagne.layers.dnn as dnn

import data
import load
import nn_plankton
import dihedral
import tmp_dnn
import tta



patch_size = (95, 95)
augmentation_params = {
    'zoom_range': (1 / 1.6, 1.6),
    'rotation_range': (0, 360),
    'shear_range': (-20, 20),
    'translation_range': (-10, 10),
    'do_flip': True,
    'allow_stretch': 1.3,
}

batch_size = 128 // 4
chunk_size = (32768//4) // 4
num_chunks_train = 840

momentum = 0.9
learning_rate_schedule = {
    0: 0.003,
    700: 0.0003,
    800: 0.00003,
}

validate_every = 1
save_every = 20


def estimate_scale(img):
    return np.maximum(img.shape[0], img.shape[1]) / 85.0
    

# augmentation_transforms_test = []
# for flip in [True, False]:
#     for zoom in [1/1.3, 1/1.2, 1/1.1, 1.0, 1.1, 1.2, 1.3]:
#         for rot in np.linspace(0.0, 360.0, 5, endpoint=False):
#             tf = data.build_augmentation_transform(zoom=(zoom, zoom), rotation=rot, flip=flip)
#             augmentation_transforms_test.append(tf)
augmentation_transforms_test = tta.build_quasirandom_transforms(70, **{
    'zoom_range': (1 / 1.4, 1.4),
    'rotation_range': (0, 360),
    'shear_range': (-10, 10),
    'translation_range': (-8, 8),
    'do_flip': True,
    'allow_stretch': 1.2,
})



data_loader = load.RescaledDataLoader(estimate_scale=estimate_scale, num_chunks_train=num_chunks_train,
    patch_size=patch_size, chunk_size=chunk_size, augmentation_params=augmentation_params,
    augmentation_transforms_test=augmentation_transforms_test)

Conv2DLayer = dnn.Conv2DDNNLayer
MaxPool2DLayer = dnn.MaxPool2DDNNLayer

#Conv2DLayer = nn.layers.dnn.Conv2DNNLayer
#MaxPool2DLayer = nn.layers.dnn.MaxPool2DNNLayer

#Conv2DLayer = tmp_dnn.Conv2DDNNLayer
#MaxPool2DLayer = tmp_dnn.MaxPool2DDNNLayer


def build_model():
    W = nn_plankton.Conv2DOrthogonal(1.0)
    b = nn.init.Constant(0.1)
    nonlin = nn_plankton.leaky_relu

    l0 = nn.layers.InputLayer((batch_size, 1, patch_size[0], patch_size[1]))
    l0c = dihedral.CyclicSliceLayer(l0)
    
    l1a_loc = MaxPool2DLayer(l0c, pool_size=(2, 2), stride=(2, 2))
    l1b_loc = Conv2DLayer(l1a_loc, num_filters=32, filter_size=(3, 3), pad="same", W=W, b=b, nonlinearity=nonlin)
    l1c_loc = Conv2DLayer(l1b_loc, num_filters=16, filter_size=(3, 3), pad="same", W=W, b=b, nonlinearity=nonlin)
    l1_loc = MaxPool2DLayer(l1c_loc, pool_size=(2, 2), stride=(2, 2))
    l1_loc_dense = nn.layers.DenseLayer(nn.layers.dropout(l1_loc, p=0.5), num_units=64, W=nn_plankton.Orthogonal(1.0), b=b, nonlinearity=nonlin)
    b1 = np.zeros((2, 3), dtype='float32')
    b1[0, 0] = 1
    b1[1, 1] = 1
    b1 = b1.flatten()
    W1 = nn.init.Constant(0.0)
    l1_loc_out = nn.layers.DenseLayer(nn.layers.dropout(l1_loc_dense, p=0.5), num_units=6, W=W1, b=b1, nonlinearity=None)
    l1_trans = nn.layers.TransformerLayer(l0c, l1_loc_out, downsample_factor=3)

    l1a = Conv2DLayer(l1_trans, num_filters=32, filter_size=(3, 3), pad="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l1b = Conv2DLayer(l1a, num_filters=16, filter_size=(3, 3), pad="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l1 = MaxPool2DLayer(l1b, pool_size=(3, 3), stride=(2, 2))
    l1r = dihedral.CyclicConvRollLayer(l1)

    l2a = Conv2DLayer(l1r, num_filters=64, filter_size=(3, 3), pad="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l2b = Conv2DLayer(l2a, num_filters=32, filter_size=(3, 3), pad="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l2 = MaxPool2DLayer(l2b, pool_size=(3, 3), stride=(2, 2))
    l2r = dihedral.CyclicConvRollLayer(l2)

    l3a = Conv2DLayer(l2r, num_filters=128, filter_size=(3, 3), pad="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l3b = Conv2DLayer(l3a, num_filters=128, filter_size=(3, 3), pad="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l3c = Conv2DLayer(l3b, num_filters=64, filter_size=(3, 3), pad="same", W=nn_plankton.Conv2DOrthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l3 = MaxPool2DLayer(l3c, pool_size=(3, 3), stride=(2, 2))
    l4r = dihedral.CyclicConvRollLayer(l3)
    l4f = nn.layers.flatten(l4r)

    l5 = nn.layers.DenseLayer(nn.layers.dropout(l4f, p=0.5), num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l5r = dihedral.CyclicRollLayer(l5)

    l6 = nn.layers.DenseLayer(nn.layers.dropout(l5r, p=0.5), num_units=256, W=nn_plankton.Orthogonal(1.0), b=nn.init.Constant(0.1), nonlinearity=nn_plankton.leaky_relu)
    l6m = dihedral.CyclicPoolLayer(l6, pool_function=nn_plankton.rms)

    l7 = nn.layers.DenseLayer(nn.layers.dropout(l6m, p=0.5), num_units=data.num_classes, nonlinearity=T.nnet.softmax, W=nn_plankton.Orthogonal(1.0))

    return [l0], l7
