# kaggle-whales
Code project for the kaggle whales competition currently on kaggle
https://www.kaggle.com/c/noaa-right-whale-recognition

The codebase used is by Sander Dieleman for his ndsb winning solution(thereof the fork, will move it to own project once I have my first top 20% submission)

Current progress:
Update to lasagne v0.2Dev - DONE
Make a data converter from kaggle whales to plankton style setup - DONE
Rewrite the data loader to not relying on whole dataset in memory - IN PROGRESS
Train a neural network - NOT STARTED

Further I have a bunch of "lead up" projects, that I will use this competition to test out, such as:

Scale SPN(code works, just needs to write config):
https://github.com/alrojo/ZoomSPN

SPN B-Trees(code works, just needs to write config):
https://github.com/alrojo/recurrent-spatial-transformer-code

Deep residual networks(Code bugs - can compile but takes 1 hr. per epoch on mnist ..!?)
https://github.com/alrojo/lasagne_residual_network
