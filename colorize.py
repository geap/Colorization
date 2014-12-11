from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
image_name = 'example.bmp'

image = misc.imread(os.path.join(dir_path, image_name))
print image
print image.shape
print image[0][0]

plt.imshow(image) #load
plt.show()  # show the window

image_name_2 = 'example_copy.bmp'
misc.imsave(os.path.join(dir_path, image_name_2), image)