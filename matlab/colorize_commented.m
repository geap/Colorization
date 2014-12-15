clear all
clc

original = double(imread('example2.bmp'))/255;
marked   = double(imread('example2_marked.bmp'))/255;
out_name='example_res_geaga.bmp';

% sum(X,dim) sums along the dimesions dim, i.e. sum(..,3) for image
% sums in the third dimension matrix columns and outputs vector with
% elements of the sums of each column
% Then takes all the colored columns as colorIm
threshold = 0.01;
colorIm = sum(abs(original - marked), 3) > threshold;
colorIm = double(colorIm);

% As the algorithm works in YUV color mode where Y is the grayscale value
% and U, V define color, rgb2ntsc is used (converts to YUV basically)
sgI=my_rgb2ntsc(original);
scI=my_rgb2ntsc(marked);

% Make a new image with Y as the grayscale first dimension
% and other dimensions from the color image

ntscIm(:,:,1)=sgI(:,:,1);
ntscIm(:,:,2)=scI(:,:,2);
ntscIm(:,:,3)=scI(:,:,3);

% Takes log from the min of the first two dimension sizes of the new image
% divides it by the log(2), substracts 2 and takes floor
% then names it "max dimension" :D
max_d=floor(log(min(size(ntscIm,1),size(ntscIm,2)))/log(2)-2);

% does some magic
iu=floor(size(ntscIm,1)/(2^(max_d-1)))*(2^(max_d-1));
ju=floor(size(ntscIm,2)/(2^(max_d-1)))*(2^(max_d-1));
id=1; jd=1;
% specifies colorIm pixels that are painted I guess
colorIm=colorIm(id:iu,jd:ju,:);
ntscIm=ntscIm(id:iu,jd:ju,:);


% solver 1 ie matlab itself by default
nI=abs(getColorExact(colorIm,ntscIm));

figure, image(nI)

% write the image to file
%imwrite(nI,out_name)
   
  

%Reminder: mex cmd
%mex -O getVolColor.cpp fmg.cpp mg.cpp  tensor2d.cpp  tensor3d.cpp
