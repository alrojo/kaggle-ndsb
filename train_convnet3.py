# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import string
import sys
from datetime import datetime, timedelta
import importlib
import time
import cPickle as pickle
from itertools import izip
import os
from subprocess import Popen

import data
import utils

if len(sys.argv) != 2:
    sys.exit("Usage: python train.py <config_name> <rmsprop/adagrad/nag>")

config_name = sys.argv[1]

config = importlib.import_module("configurations.%s" % config_name)
optimizer = config.optimizer
print "Using configurations: '%s'" % config_name

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (config_name, timestamp)
metadata_tmp_path = "/var/tmp/%s.pkl" % experiment_id
metadata_target_path = os.path.join(os.getcwd(), "metadata/%s.pkl" % experiment_id)

print "Experiment id: %s" % experiment_id

sym_y = T.imatrix('target_output')
sym_x = T.tensor3()
TOL = 1e-8
num_epochs = config.epochs
batch_size = config.batch_size
num_classes = data.num_classes
chunk_size = config.chunk_size

print("Building network ...")
l_in, l_out = config.build_model()

all_layers = nn.layers.get_all_layers(l_out)
num_params = nn.layers.count_params(l_out)

print("  number of parameters: %d" % num_params)
print("  layer output shapes:")
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    print("    %s %s" % (name, nn.layers.get_output_shape(layer)))
print("Creating cost function")
out_train = nn.layers.get_output(
    l_out, sym_x, deterministic=False)
out_eval = nn.layers.get_output(
    l_out, sym_x, deterministic=True)
probs_flat = out_train.reshape((-1, num_classes))

lambda_reg = config.lambda_reg
params = nn.layers.get_all_params(l_out, regularizable=True)
reg_term = sum(T.sum(p**2) for p in params)
cost = T.nnet.categorical_crossentropy(T.clip(probs_flat, TOL, 1-TOL), sym_y.flatten())
cost = T.sum(cost) + lambda_reg * reg_term
all_params = nn.layers.get_all_params(l_out, trainable=True)

if hasattr(config, 'set_weights'):
    nn.layers.set_all_param_values(l_out, config.set_weights())

print("Computing updates ...")
if hasattr(config, 'learning_rate_schedule'):
    learning_rate_schedule = config.learning_rate_schedule              # Import learning rate schedule
else:
    learning_rate_schedule = { 0: config.learning_rate }
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

all_grads = T.grad(cost, all_params)

cut_norm = config.cut_grad
updates, norm_calc = nn.updates.total_norm_constraint(all_grads, max_norm=cut_norm, return_norm=True)

momentum_schedule = config.momentum_schedule
momentum = theano.shared(np.float32(momentum_schedule[0]))
updates = nn.updates.nesterov_momentum(updates, all_params, learning_rate, momentum)

print "config.batch_size %d" % batch_size
print "data.num_classes %d" % num_classes
print("Compiling functions ...")

train = theano.function(
    [sym_x, sym_y], [cost, out_train, norm_calc], updates=updates)
eval = theano.function([sym_x], [out_eval])

start_time = time.time()
prev_time = start_time

all_losses_train = []
all_accuracy_train = []
all_losses_eval_train = []
all_losses_eval_valid = []
all_losses_eval_test = []
all_accuracy_eval_train = []
all_accuracy_eval_valid = []
all_accuracy_eval_test = []
all_mean_norm = []


create_train_gen = lambda: config.data_loader.create_random_gen(config.data_loader.images_train, config.data_loader.labels_train)
create_eval_valid_gen = lambda: config.data_loader.create_fixed_gen(config.data_loader.images_valid, augment=False)
create_eval_train_gen = lambda: config.data_loader.create_fixed_gen(config.data_loader.images_train, augment=False)

num_batches_chunk = chunk_size // batch_size

copy_process = None

print "It has begun ..."
for epoch, (X_chunk_list, y_chunk) in izip(np.arange(0, num_epochs), create_train_gen()):
    X_chunk = np.array(X_chunk_list, dtype='float32').squeeze() # hacking the multiscale away
    print "Epoch %d of %d" % (epoch + 1, num_epochs)

    if epoch in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[epoch])
        print "  setting learning rate to %.7f" % lr
        learning_rate.set_value(lr)
    if epoch in momentum_schedule:
        mu = np.float32(momentum_schedule[epoch])
        print "  setting learning rate to %.7f" % mu
        momentum.set_value(mu)

    print "  batch SGD"
    losses = []
    preds = []
    norms = []
    for i in range(num_batches_chunk):
        idx = range(i*batch_size, (i+1)*batch_size)
        X_batch = X_chunk[idx]
        y_batch = y_chunk[idx]
        loss, out, batch_norm = train(X_batch, y_batch)
        print(batch_norm)
        norms.append(batch_norm)
        preds.append(out)
        losses.append(loss)

    predictions = np.concatenate(preds, axis = 0)
    loss_train = np.mean(losses)
    all_losses_train.append(loss_train)

    mean_norm = np.mean(norms)
    all_mean_norm.append(mean_norm)

    print "  average training loss: %.5f" % loss_train


    if ((epoch) % config.validate_every) == 0:
        print
        print "Validating"
        subsets = ["train", "valid"]
        gens = [create_eval_train_gen, create_eval_valid_gen]
        label_sets = [config.data_loader.labels_train, config.data_loader.labels_valid]
        losses_eval = [all_losses_eval_train, all_losses_eval_valid]

        for subset, create_gen, labels, losses in zip(subsets, gens, label_sets, losses_eval):
            print "  %s set" % subset
            outputs = []
            for X_chunk_eval_list, chunk_length_eval in create_gen():
                X_chunk_eval = np.asarray(X_chunk_eval_list, dtype='float32').squeeze()
                num_batches_chunk_eval = int(np.ceil(chunk_length_eval / float(config.batch_size)))
                
                outputs_chunk = []
                
                for i in range(num_batches_chunk_eval):
                    idx = range(i*batch_size, (i+1)*batch_size)
                    X_batch_eval = X_chunk_eval[idx]
                    out = eval(X_batch_eval)
                    outputs_chunk.append(out)
                
                outputs_chunk.append
                outputs_chunk = np.vstack(outputs_chunk)
                outputs_chunk = outputs_chunk[:chunk_length_eval] # truncate to the right length
                outputs.append(outputs_chunk)
                
            outputs = np.vstack(outputs)
            loss = utils.log_loss(outputs, labels)
            acc = utils.accuracy(outputs, labels)
            print "    loss:\t%.6f" % loss
            print "    acc:\t%.2f%%" % (acc * 100)
            print
            
            losses.append(loss)
            del outputs

    now = time.time()
    time_since_start = now - start_time
    time_since_prev = now - prev_time
    prev_time = now
    est_time_left = time_since_start * (float(config.num_chunks_train - (epoch + 1)) / float(epoch + 1))
    eta = datetime.now() + timedelta(seconds=est_time_left)
    eta_str = eta.strftime("%c")
    print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
    print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
    print

    if ((epoch + 1) % config.save_every) == 0:
        print
        print "Saving metadata, parameters"

        with open(metadata_tmp_path, 'w') as f:
            pickle.dump({
                'configuration': config_name,
                'experiment_id': experiment_id,
                'chunks_since_start': epoch,
                'losses_train': all_losses_train,
                'losses_eval_valid': all_losses_eval_valid,
                'losses_eval_train': all_losses_eval_train,
                'time_since_start': time_since_start,
                'param_values': nn.layers.get_all_param_values(l_out), 
                'data_loader_params': config.data_loader.get_params(),
            }, f, pickle.HIGHEST_PROTOCOL)

        # terminate the previous copy operation if it hasn't finished
        if copy_process is not None:
            copy_process.terminate()

        copy_process = Popen(['cp', metadata_tmp_path, metadata_target_path])

        print "  saved to %s, copying to %s" % (metadata_tmp_path, metadata_target_path)
        print