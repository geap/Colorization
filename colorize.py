from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

original = misc.imread(os.path.join(dir_path, 'example.bmp'))
marked = misc.imread(os.path.join(dir_path, 'example_marked.bmp'))

original = original.astype(float)/255
marked = marked.astype(float)/255

"""
print 'image'
print image
print 'shape'
print image.shape
print 'esimese pixli rgb'
print image[0][0]

plt.imshow(image) #load
plt.show()  # show the window
"""



#image_name_2 = 'example_copy.bmp'
#misc.imsave(os.path.join(dir_path, image_name_2), image)