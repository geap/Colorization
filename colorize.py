from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import os
import time
import sys
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy import io

os.system('cls')
os.system('reset')

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
    label_inds = np.count_nonzero(color_copy_for_nonzero) # it's cool that nonzero likes boolean values, too
    
    wd = 1
    
    # length as len (for obv reasons)
    length = 0
    col_inds = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    row_inds = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    vals = np.zeros((image_size*( 2 * wd + 1 )**2,1))
    gvals = np.zeros((2 * wd + 1 )**2)
    
    
    # PREPS made, lets ITERATE!
    count = 0 # for testing
    consts_len = 0
    for j in range(m):
        for i in range(n):
            consts_len += 1
            
            if (not colorIm[i,j]):
                tlen = 0
                
                #print max( 0, i - wd ), min( i + wd+1, n ),max( 0, j - wd ), min( j + wd, m )+1
                for ii in range(max( 0, i-wd ), min( i+wd+1, n )):
                    for jj in range( max( 0, j - wd ), min( j + wd, m )+1):
                        count += 1 # for testing
                        if ( ii != i or jj != j ):
                            row_inds[length,0] = consts_len
                            col_inds[length,0] = indices_matrix[ii,jj]
                            gvals[tlen] = YUV[ii,jj,0]
                            length += 1
                            tlen += 1
                
                t_val = YUV[i,j,0].copy()
                gvals[tlen] = t_val
                c_var = np.mean((gvals[0:tlen+1] - np.mean(gvals[0:tlen+1]))**2)
                csig = c_var * 0.6
                mgv = min(( gvals[0:tlen+1] - t_val )**2)
                
                if (csig < ( -mgv / np.log(0.01 ))):
                    csig = -mgv / np.log(0.01)
                if (csig <0.000002):
                    csig = 0.000002
                
                gvals[0:tlen] = np.exp( -(gvals[0:tlen] - t_val)**2 / csig )
                gvals[0:tlen] = gvals[0:tlen] / np.sum(gvals[0:tlen])
                vals[length-tlen:length,0] = -gvals[0:tlen]
            
            # END IF
            
            length += 1
            row_inds[length-1,0] = consts_len
            col_inds[length-1,0] = indices_matrix[i,j]
            vals[length-1,0] = 1
            
            
    
        # END OF FOR i
    # END OF FOR j
    
    
    # A LITTLE BIT MORE AND THEN CAN RETURN ALREADY SOMETHING!
    
    vals = vals[0:length,0]
    col_inds = col_inds[0:length,0]
    row_inds = row_inds[0:length,0]
    
    # A=sparse(row_inds,col_inds,vals,consts_len,imgSize);
    
    '''THOUGHT FOOD
    S = sparse(i,j,s,m,n,nzmax) uses vectors i, j, and s to generate an
    m-by-n sparse matrix such that S(i(k),j(k)) = s(k), with space
    allocated for nzmax nonzeros.  Vectors i, j, and s are all the same
    length.  Any elements of s that are zero are ignored, along with the
    corresponding values of i and j.  Any elements of s that have duplicate
    values of i and j are added together.  The argument s and one of the
    arguments i or j may be scalars, in which case the scalars are expanded
    so that the first three arguments all have the same length.
    
    >> a = diag(1:4)
    
    a =
    
        1     0     0     0
        0     2     0     0
        0     0     3     0
        0     0     0     4
    
    >> s = sparse(a)
    
    s =
    
    (1,1)        1
    (2,2)        2
    (3,3)        3
    (4,4)        4
   '''
    
    
    #print something
    sys.exit('Sparse needs to be implemented!')
    
    return YUV # should be colorized, but mock until we make it

# read in grayscale and marked image
dir_path = os.path.dirname(os.path.realpath(__file__))
original = misc.imread(os.path.join(dir_path, 'example.bmp'))
marked = misc.imread(os.path.join(dir_path, 'example_marked.bmp'))

original = original.astype(float)/255
marked = marked.astype(float)/255

# colorIm as isColored
# calculate where colors are given
isColored = abs(original - marked).sum(2) > 0.01

# convert the image from RGB to YIQ
# luma component does not change, so the original image can be used for calculations
(Y,_,_) = colorsys.rgb_to_yiq(original[:,:,0],original[:,:,1],original[:,:,2])
# calculate chromimance components
(_,I,Q) = colorsys.rgb_to_yiq(marked[:,:,0],marked[:,:,1],marked[:,:,2])

# YUV aka ntscIm
YUV = np.zeros(original.shape)
YUV[:,:,0] = Y
YUV[:,:,1] = I
YUV[:,:,2] = Q

'''
max_d = np.floor(np.log(min(YUV.shape[0],YUV.shape[1]))/np.log(2)-2)
iu = np.floor(YUV.shape[0]/(2**(max_d - 1))) * (2**(max_d - 1))
ju = np.floor(YUV.shape[1]/(2**(max_d - 1))) * (2**(max_d - 1))
colorIm = colorIm[:iu,:ju]
YUV = YUV[:iu,:ju]
'''

# SOLVE THIS PROBLEM
#colorized = abs(getColorExact( colorIm, YUV ));

############## getColorExact Start ################
#def getColorExact( colorIm, YUV):
# YUV as ntscIm

n = YUV.shape[0] # image height
m = YUV.shape[1] # image width
image_size = n*m

# colorized as nI
# variable for the colorized result in YIQ
colorized = np.zeros(YUV.shape)
# luma component stays the same
colorized[:,:,0] = YUV[:,:,0]

# this Matlab code: z = reshape(x,3,4); should become z = x.reshape(3,4,order='F').copy() in Numpy.
# indices_matrix as indsM
# enumerate indices
indices_matrix = np.arange(image_size).reshape(n,m,order='F').copy()

# the radius of window around the pixel to assess
wd = 1
# the number of pixels in the window
nr = (2*wd + 1)**2
# maximal size of pixels to assess for the hole image
max_nr = image_size * nr

# set up variables for row indices, column indices, values and window values
row_inds = np.zeros((max_nr, 1), dtype=np.int64)
col_inds = np.zeros((max_nr, 1), dtype=np.int64)
vals = np.zeros((max_nr, 1))
# gvals as window_vals
window_vals = np.zeros(nr)

# PREPS made, lets ITERATE!
length = 0
consts_len = 0
# iterate over pixels in the image
for j in range(m):
    for i in range(n):
        consts_len += 1
        
        if (not isColored[i,j]): # the pixel is not already colored
            # tlen as window_index
            window_index = 0
            
            # iterate over pixels in the window with the center [i,j]
            for ii in range(max(0, i-wd), min(i+wd+1,n)): # min(i+wd,n) -> min( i+wd+1, n )
                for jj in range(max(0, j-wd), min(j+wd, m) + 1): # but min(j+wd,m) -> min( j + wd, m )+1
                    if (ii != i or jj != j): # not the center pixel
                        row_inds[length,0] = consts_len
                        col_inds[length,0] = indices_matrix[ii,jj]
                        window_vals[window_index] = YUV[ii,jj,0]
                        length += 1
                        window_index += 1
            
            # t_val as center
            center = YUV[i,j,0].copy()
            window_vals[window_index] = center
            
            # calculate variance of the intensities in a window around pixel [i,j]
            # c_var as variance
            variance = np.mean((window_vals[0:window_index+1] - np.mean(window_vals[0:window_index+1]))**2)
            #csig as sigma
            sigma = variance * 0.6 # don't really understand why this is necessary... based on article I would multiply with 2
            
            # magic
            mgv = min(( window_vals[0:window_index+1] - center )**2)            
            if (sigma < ( -mgv / np.log(0.01 ))):
                sigma = -mgv / np.log(0.01)
            # avoid dividing by 0
            if (sigma < 0.000002):
                sigma = 0.000002
            
            # use weighting funtion (2)
            window_vals[0:window_index] = np.exp( -((window_vals[0:window_index] - center)**2) / sigma )
            
            # make the weights sum up to 1
            window_vals[0:window_index] = window_vals[0:window_index] / np.sum(window_vals[0:window_index])
            
            # weights calculated
            vals[length-window_index:length,0] = -window_vals[0:window_index]
        
        # END IF
        
        length += 1
        row_inds[length-1,0] = consts_len
        col_inds[length-1,0] = indices_matrix[i,j]
        vals[length-1,0] = 1
        
        

    # END OF FOR i
# END OF FOR j


# A LITTLE BIT MORE AND THEN CAN RETURN ALREADY SOMETHING!

vals = vals[0:length,0]
col_inds = col_inds[0:length,0]
row_inds = row_inds[0:length,0]

# decrease indexes by 1 because somehow sparse increases indexes by 1
#col_inds[col_inds > 0] = col_inds[col_inds > 0] - 1
#row_inds[row_inds > 0] = row_inds[row_inds > 0] - 1

# otherwise index out of bounds
if (row_inds[row_inds.shape[0]-1] == consts_len):
    row_inds[row_inds.shape[0]-1] = consts_len - 1

# A=sparse(row_inds,col_inds,vals,consts_len,imgSize);
    
'''
S = sparse(i,j,s,m,n,nzmax) uses vectors i, j, and s to generate an
m-by-n sparse matrix such that S(i(k),j(k)) = s(k), with space
allocated for nzmax nonzeros.  Vectors i, j, and s are all the same
length.  Any elements of s that are zero are ignored, along with the
corresponding values of i and j.  Any elements of s that have duplicate
values of i and j are added together.  The argument s and one of the
arguments i or j may be scalars, in which case the scalars are expanded
so that the first three arguments all have the same length.

>> a = diag(1:4)

a =

    1     0     0     0
    0     2     0     0
    0     0     3     0
    0     0     0     4

>> s = sparse(a)

s =

(1,1)        1
(2,2)        2
(3,3)        3
(4,4)        4
'''


# csr_matrix((data, ij), [shape=(M, N)])
# where data and ij satisfy the relationship a[ij[0, k], ij[1, k]] = data[k]

# make the sparse matrix
A = sps.coo_matrix((vals, (row_inds, col_inds)), (consts_len, image_size))
#A_dense = A.todense() # MemoryError
# write to a file instead
io.mmwrite(os.path.join(dir_path, 'sparse_matrix'), A)
# for some unknown reason all the coordinates are increased by 1
print "row_inds"
print row_inds[0:10]
print "col_inds"
print col_inds[0:10]
print "vals"
print vals[0:10]

# The CSR format is specially suitable for fast matrix vector product
# Try first with the same format as in the example in http://docs.scipy.org/doc/scipy-0.14.0/reference/sparse.html
A = A.tolil().tocsr()
#io.mmwrite(os.path.join(dir_path, 'sparse_matrix_lil'), A)

print 'Get the sparse matrix correct and then continue :)'

# We have to reshape and make a copy of the view of an array
# for the nonzero() work like in MATLAB
color_copy_for_nonzero = isColored.reshape(image_size).copy()

# label_inds as lblInds
#label_inds = np.count_nonzero(color_copy_for_nonzero) # it's cool that nonzero likes boolean values, too
# the indeces not the count of them is needed
label_inds = np.nonzero(color_copy_for_nonzero) # it's cool that nonzero likes boolean values, too

#b = np.zeros((A.shape[0], 1))
#curIm = np.zeros(YUV.shape)
for t in [1,2]: # not [2,3] due to different indexing 
    print t
    curIm = YUV[:,:,t]
    curIm = curIm.reshape(image_size, 1) # because label_inds are up to 84800
    b = np.zeros((image_size, 1))
    b[label_inds] = curIm[label_inds]
    #new_vals = np.divide(A, b) # very, very slow
    new_vals = spsolve(A, b) # this should work, see examples from http://docs.scipy.org/doc/scipy-0.14.0/reference/sparse.html
    # sps.linalg.inv(A) # A is singular, there is no inverse matrix
    #new_vals = sps.linalg.lsqr(A, b) # this works endlessly...
    print "new vals"
    print new_vals
    colorized[:,:,t] = new_vals.reshape(n, m, order='F').copy()

#snI = colorized

#(R, G, B) = colorsys.yiq_to_rgb(colorized[:,:,0],colorized[:,:,1],colorized[:,:,2])

#colorizedIm = np.zeros(original.shape)
#colorizedIm[:,:,0] = R
#colorizedIm[:,:,1] = G
#colorizedIm[:,:,2] = B


#print something
#sys.exit('Sparse needs to be implemented!')

#scipy.sparse.csr_matrix

    
#    return YUV # should be colorized, but mock until we make it

################## getColorExact end ##############


#plt.imshow(colorized)
#plt.show()

#misc.imsave(os.path.join(dir_path, 'example_colorized.bmp'), colorizedIm)
