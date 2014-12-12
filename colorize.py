from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
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
    indices_matrix = np.arange(image_size).reshape(n,m,order='F').copy()
    
    # We have to reshape and make a copy of the view of an array
    # for the nonzero() work like in MATLAB
    color_copy_for_nonzero = colorIm.reshape(image_size).copy()
    # label_inds as lblInds
    label_inds = np.nonzero(color_copy_for_nonzero) # it's cool that nonzero likes boolean values, too
    
    wd = 1
    
    # length as len (for obv reasons)
    length = -1
    col_inds = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    row_inds = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    vals = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    gvals = np.zeros((1,(2 * wd + 1 )**2))    
    
    
    # PREPS made, lets ITERATE!
    
    consts_len = -1
    for j in range(m):
        for i in range(n):
            consts_len += 1
            
            if (not colorIm[i,j]):
                tlen = -1
                
                for ii in range(max( 0, i - wd ), min( i + wd, n ) +1):
                    for jj in range( max( 0, j - wd ), min( j + wd, m ) +1):
                        if ( ii != i or jj != j ):
                            length += 1
                            tlen += 1
                            row_inds[length,0] = consts_len
                            col_inds[length,0] = indices_matrix[ii,jj]
                            gvals[0,tlen] = YUV[ii,jj,0]
                
                t_val = YUV[i,j,0]
                gvals[0,tlen+1] = t_val
                c_var = np.mean((gvals[0,0:tlen+1]-np.mean(gvals[0,0:tlen+1]))**2)
                csig = c_var * 0.6
                mgv = min(( gvals[0,0:tlen] - t_val )**2)
                
                if (csig < ( -mgv / np.log(0.01 ))):
                    csig = -mgv / np.log(0.01)
                if (csig <0.000002):
                    csig = 0.000002
                
                gvals[0,0:tlen] = np.exp( -(gvals[0:tlen] - t_val)**2 / csig )
                gvals[0,0:tlen] = gvals[0,0:tlen] / np.sum(gvals[0,0:tlen])
                vals[length-tlen:length,0] = -gvals[0,0:tlen]
            
            # END IF
            
            length += 1
            row_inds[length,0] = consts_len
            col_inds[length,0] = indices_matrix[i,j]
            vals[length,0] = 1
        
        # END OF FOR i
    # END OF FOR j
    
    # A LITTLE BIT MORE AND THEN CAN RETURN ALREADY SOMETHING!
    
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


# SOLVE THIS PROBLEM
colorized = abs(getColorExact( colorIm[:iu,:ju], YUV[:iu,:ju] ));


#plt.imshow(colorized)
#plt.show()

#misc.imsave(os.path.join(dir_path, 'example_colorized.bmp'), image)
