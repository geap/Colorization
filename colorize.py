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

# http://samarthbhargav.wordpress.com/2014/05/05/image-processing-with-python-rgb-to-grayscale-conversion/

plt.imshow(image) #load
plt.show()  # show the window

#...

image_name_2 = 'example_copy.bmp'
misc.imsave(os.path.join(dir_path, image_name_2), image)
