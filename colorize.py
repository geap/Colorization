from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import os

def getColorExact( colorIm, YUV):
    # YUV as ntscIm
    
    n = YUV.shape[0]
    m = YUV.shape[1]
    
    image_size = n*m
    
    # colorized as nI
    colorized = np.zeros(YUV.shape)
    colorized[:,:,0] = YUV[:,:,0]
    
    
    # this Matlab code: z = reshape(x,3,4); should become z = x.reshape(3,4,order='F').copy() in Numpy.
    # indices_matrix as indsM
    indices_matrix = np.arange(image_size).reshape(n,m,order='F')
    # label_inds as lblInds
    label_inds = np.nonzero(indices_matrix)
    
    wd = 1
    
    # length as len (for obv reasons)
    length = 0
    consts_len = 0
    col_inds = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    row_inds = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    vals = np.zeros(image_size*( 2 * wd + 1 )**2,1)
    gvals = np.zeros((1,2 * wd + 1 )**2);
    
    return YUV # should be colorized, but mock until we make it




dir_path = os.path.dirname(os.path.realpath(__file__))

original = misc.imread(os.path.join(dir_path, 'example.bmp'))
marked = misc.imread(os.path.join(dir_path, 'example_marked.bmp'))

original = original.astype(float)/255
marked = marked.astype(float)/255

# values on True/False not 1/0 as in MATLAB - deal later with it!
colorIm = abs(original - marked).sum(2) > 0.01

(Y,_,_) = colorsys.rgb_to_yiq(original[:,:,0],original[:,:,1],original[:,:,2])
(_,I,Q) = colorsys.rgb_to_yiq(marked[:,:,0],marked[:,:,1],marked[:,:,2])

# YUV aka ntscIm
YUV = np.zeros(original.shape)
YUV[:,:,0] = Y
YUV[:,:,1] = I
YUV[:,:,2] = Q

max_d = np.floor(np.log(min(YUV.shape[0],YUV.shape[1]))/np.log(2)-2)

iu = np.floor(YUV.shape[0]/(2**(max_d - 1))) * (2**(max_d - 1))
ju = np.floor(YUV.shape[1]/(2**(max_d - 1))) * (2**(max_d - 1))

print colorIm.shape
colorIm = colorIm[:iu,:ju]
YUV = YUV[:iu,:ju]


# SOLVE THIS PROBLEM
colorized = abs(getColorExact( colorIm, YUV ));


plt.imshow(colorized)
plt.show()

#misc.imsave(os.path.join(dir_path, 'example_colorized.bmp'), image)
