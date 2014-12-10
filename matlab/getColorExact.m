function [nI,snI]=getColorExact(colorIm,YUV)

n=size(YUV,1); m=size(YUV,2);
imgSize=n*m;


nI(:,:,1) = YUV(:,:,1);

% reshape(X,M,N) returns the M-by-N matrix whose elements
% are taken columnwise from X.
indsM=reshape([1:imgSize],n,m);
% I = find(X) returns the linear indices corresponding to 
%     the nonzero entries of the array X.  X may be a logical expression. 
color_label_linear_indices=find(colorIm);

weight = 1; 

len=0;
consts_len=0;
col_inds=zeros(imgSize*(2*weight+1)^2,1);
row_inds=zeros(imgSize*(2*weight+1)^2,1);
vals=zeros(imgSize*(2*weight+1)^2,1);
gvals=zeros(1,(2*weight+1)^2);


for j=1:m
   for i=1:n
      consts_len=consts_len+1;
      
      if (~colorIm(i,j))   
        tlen=0;
        for ii=max(1,i-weight):min(i+weight,n)
           for jj=max(1,j-weight):min(j+weight,m)
            
              if (ii~=i)|(jj~=j)
                 len=len+1; tlen=tlen+1;
                 row_inds(len)= consts_len;
                 col_inds(len)=indsM(ii,jj);
                 gvals(tlen)=YUV(ii,jj,1);
              end
           end
        end
        t_val=YUV(i,j,1);
        gvals(tlen+1)=t_val;
%         For vectors, mean(X) is the mean value of the elements in X. For
%     matrices, mean(X) is a row vector containing the mean value of
%     each column.  For N-D arrays, mean(X) is the mean value of the
%     elements along the first non-singleton dimension of X.
        c_var=mean((gvals(1:tlen+1)-mean(gvals(1:tlen+1))).^2);
        csig=c_var*0.6;
        mgv=min((gvals(1:tlen)-t_val).^2);
        if (csig<(-mgv/log(0.01)))
	   csig=-mgv/log(0.01);
	end
	if (csig<0.000002)
	   csig=0.000002;
        end
% exp(X) is the exponential of the elements of X, e to the X.
        gvals(1:tlen)=exp(-(gvals(1:tlen)-t_val).^2/csig);
        gvals(1:tlen)=gvals(1:tlen)/sum(gvals(1:tlen));
        vals(len-tlen+1:len)=-gvals(1:tlen);
      end

        
      len = len+1;
      row_inds(len) = consts_len;
      col_inds(len) = indsM(i,j);
      vals(len) = 1; 

   end
end

       
vals = vals(1:len);
col_inds = col_inds(1:len);
row_inds = row_inds(1:len);

% S = sparse(X) converts a sparse or full matrix to sparse form by squeezing
%     out any zero elements.
%  
%     S = sparse(i,j,s,m,n,nzmax) uses vectors i, j, and s to generate an
%     m-by-n sparse matrix such that S(i(k),j(k)) = s(k), with space
%     allocated for nzmax nonzeros.  Vectors i, j, and s are all the same
%     length.  Any elements of s that are zero are ignored, along with the
%     corresponding values of i and j.  Any elements of s that have duplicate
%     values of i and j are added together.  The argument s and one of the
%     arguments i or j may be scalars, in which case the scalars are expanded
%     so that the first three arguments all have the same length.
A=sparse(row_inds,col_inds,vals,consts_len,imgSize);
b=zeros(size(A,1),1);


for t = 2:3
    curIm = YUV(:,:,t);
    b(color_label_linear_indices) = curIm(color_label_linear_indices);
    
%     A\B is the matrix division of A into B, which is roughly the
%     same as INV(A)*B , except it is computed in a different way.
%     If A is an N-by-N matrix and B is a column vector with N
%     components, or a matrix with several such columns, then
%     X = A\B is the solution to the equation A*X = B.
    new_vals = A\b;
    
%     reshape(X,M,N) returns the M-by-N matrix whose elements
%     are taken columnwise from X. 
%     reshape(X,M,N,P,...) returns an N-D array with the same
%     elements as X but reshaped to have the size M-by-N-by-P-by-...
    nI(:,:,t) = reshape(new_vals,n,m,1);    
end



snI = nI;
nI = my_ntsc2rgb(nI);

