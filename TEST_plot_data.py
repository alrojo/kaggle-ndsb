# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

dat = np.load('shapes.npy')

plt.plot(dat[:,0], dat[:,1], '.')
plt.show