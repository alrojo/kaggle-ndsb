import numpy as np
import utils
import matplotlib.pyplot as plt

pics = utils.load_gz("test_time_pics.npy.gz")
print pics.shape
pic_list = [slice(pics, idx, axis=0) for idx in range(pics.shape[0])]
plt.plot(pic_list[0])
plt.plot(pic_list[20])
plt.plot(pic_list[-1])
plt.show
