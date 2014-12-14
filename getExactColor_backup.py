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
