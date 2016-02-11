# kaggle-whales
Code project for the kaggle whales competition on kaggle
https://www.kaggle.com/c/noaa-right-whale-recognition

The codebase used is by Sander Dieleman for his ndsb winning solution.

## Current progress

Update to lasagne v0.2Dev - DONE

Make a data converter from kaggle whales to plankton style setup - DONE

Rewrite the data loader to not relying on whole dataset in memory - DONE (though as I only use 512x512 it can actually fit in memory, why I use the branch testingreg now)

Train a neural network - Got a top 20% submission

## Supporting projects

Further I have a bunch of "lead up" projects, that I havn't had time to test out, but could be cool.

Scale SPN(code works, just needs to write config):
https://github.com/alrojo/ZoomSPN

SPN B-Trees(code works, just needs to write config):
https://github.com/alrojo/recurrent-spatial-transformer-code

Deep residual networks(code works, just needs to write config)
https://github.com/alrojo/lasagne_residual_network

## Other
Should test out ZCA-whitening of local pixel chunks, like 8x8 or 16x16.
