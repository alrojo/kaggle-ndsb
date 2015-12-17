import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

pics = np.load("chunk_x.npy")
print pics.shape
pic_list = []
for i in range(pics.shape[0]):
    pic_list.append(pics[i])
plt.imshow(pic_list[641], cmap = cm.Greys_r)
plt.plot(pic_list[20])
plt.plot(pic_list[-1])
plt.show