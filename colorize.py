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


# ---------------------------------------------------------------------------- #
# ------------------------------- PREPARE ------------------------------------ #
# ---------------------------------------------------------------------------- #

dir_path = os.path.dirname(os.path.realpath(__file__))
original = misc.imread(os.path.join(dir_path, 'example.bmp'))
marked = misc.imread(os.path.join(dir_path, 'example_marked.bmp'))

original = original.astype(float)/255
marked = marked.astype(float)/255

isColored = abs(original - marked).sum(2) > 0.01                                # isColored as colorIm 

(Y,_,_) = colorsys.rgb_to_yiq(original[:,:,0],original[:,:,1],original[:,:,2])
(_,I,Q) = colorsys.rgb_to_yiq(marked[:,:,0],marked[:,:,1],marked[:,:,2])

YUV = np.zeros(original.shape)                                                  # YUV as ntscIm
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
                                                                                # ALTERNATIVE :: colorized = abs(getColorExact( colorIm, YUV ));

# ---------------------------------------------------------------------------- #
# ---------------------------- getExactColor --------------------------------- #
# ---------------------------------------------------------------------------- #

                                                                                # YUV as ntscIm
n = YUV.shape[0]                                                                # n = image height
m = YUV.shape[1]                                                                # m = image width
image_size = n*m


colorized = np.zeros(YUV.shape)                                                 # colorized as nI = resultant colored image
colorized[:,:,0] = YUV[:,:,0]


indices_matrix = np.arange(image_size).reshape(n,m,order='F').copy()            # indices_matrix as indsM

wd = 1                                                                          # The radius of window around the pixel to assess

nr = (2*wd + 1)**2                                                              # The number of pixels in the window
max_nr = image_size * nr                                                        # Maximal size of pixels to assess for the hole image
                                                                                # (for now include the full window also for the border pixels)

row_inds = np.zeros((max_nr, 1), dtype=np.int64)
col_inds = np.zeros((max_nr, 1), dtype=np.int64)
vals = np.zeros((max_nr, 1))
                                                                                # window_vals as gvals 
                                                                                # added this inside the loop
                                                                                # window_vals = np.zeros(nr) 

# ----------------------------- Interation ----------------------------------- #

length = 0
pixel_nr = 0                                                                    # the nr of the current pixel, this corresponds to the row index in sparse matrix

for j in range(m):                                                             # iterate over pixels in the image
    for i in range(n):
        #pixel_nr += 1  IS commented out, because this way there won't be row index is sparse matrix with 0, added this to end 
        
        if (not isColored[i,j]): # The pixel is not colored yet
            
            window_index = 0                                                    # tlen as window_index
            window_vals = np.zeros(nr)
                                                                                # iterate over pixels in the window with the center [i,j]
            for ii in range(max(0, i-wd), min(i+wd+1,n)):                       # CHECK: min(i+wd,n) -> min( i+wd+1, n )
                for jj in range(max(0, j-wd), min(j+wd, m) + 1):                # CHECK: but min(j+wd,m) -> min( j + wd, m )+1
                    if (ii != i or jj != j):                                    # not the center pixel
                        row_inds[length,0] = pixel_nr
                        col_inds[length,0] = indices_matrix[ii,jj]
                        window_vals[window_index] = YUV[ii,jj,0]
                        length += 1
                        window_index += 1
            
            center = YUV[i,j,0].copy()                                          # t_val as center
            window_vals[window_index] = center
            
            # calculate variance of the intensities in a window around pixel [i,j]
            # c_var as variance
            variance = np.mean((window_vals[0:window_index+1] - np.mean(window_vals[0:window_index+1]))**2)
            
            sigma = variance * 0.6                                              #csig as sigma
            
            mgv = min(( window_vals[0:window_index+1] - center )**2)            
            if (sigma < ( -mgv / np.log(0.01 ))):
                sigma = -mgv / np.log(0.01)                                     # avoid dividing by 0
            if (sigma < 0.000002):
                sigma = 0.000002
            
            # use weighting funtion (2)
            window_vals[0:window_index] = np.exp( -((window_vals[0:window_index] - center)**2) / sigma )
            
            # make the weighting function sum up to 1
            window_vals[0:window_index] = window_vals[0:window_index] / np.sum(window_vals[0:window_index])
            
            # add calculated weights
            vals[length-window_index:length,0] = -window_vals[0:window_index]
        
        # END IF
        
        # add the values for the current pixel
        row_inds[length,0] = pixel_nr
        col_inds[length,0] = indices_matrix[i,j]
        vals[length,0] = 1
        length += 1
        pixel_nr += 1
        
    # END OF FOR i
# END OF FOR j

print row_inds.shape
print col_inds.shape
print vals.shape
print window_vals
sys.exit('ERROR')

# ------------------------ After Iteration Process --------------------------- #
# ------------------------ After Iteration Process --------------------------- #
# ------------------------ After Iteration Process --------------------------- #

# trim to variables to the actually used length that does not include the full window for the border pixels
vals = vals[0:length,0]
col_inds = col_inds[0:length,0]
row_inds = row_inds[0:length,0]


# ------------------------------- Sparseness --------------------------------- #

# decrease indexes by 1 because somehow sparse increases indexes by 1
#col_inds[col_inds > 0] = col_inds[col_inds > 0] - 1
#row_inds[row_inds > 0] = row_inds[row_inds > 0] - 1

# A=sparse(row_inds,col_inds,vals,consts_len,imgSize);
# csr_matrix((data, ij), [shape=(M, N)])
# where data and ij satisfy the relationship a[ij[0, k], ij[1, k]] = data[k]

# make the sparse matrix
A = sps.coo_matrix((vals, (row_inds, col_inds)), (pixel_nr, image_size))
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
