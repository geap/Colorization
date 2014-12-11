from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

original = misc.imread(os.path.join(dir_path, 'example.bmp'))
marked = misc.imread(os.path.join(dir_path, 'example_marked.bmp'))

original = original.astype(float)/255
marked = marked.astype(float)/255

# values on True/False not 1/0 as in MATLAB - deal later with it!
colorIm = abs(original - marked).sum(2) > 0.01

(Y,I,Q) = colorsys.rgb_to_yiq(original.take((0,),2),original.take((1,),2),original.take((2,),2))

yiq_gray = np.zeros(original.shape)
yiq_gray[:][:][0] = Y
yiq_gray[:][:][1] = I
yiq_gray[:][:][2] = Q

"""
YIQ_gray = my_rgb2ntsc(original);
YIQ_color = my_rgb2ntsc(marked);
print 'image'
print image
print image.shape
print image[0][0]
plt.imshow(image) #load
plt.show()  # show the window
"""

#image_name_2 = 'example_copy.bmp'
#misc.imsave(os.path.join(dir_path, image_name_2), image)
