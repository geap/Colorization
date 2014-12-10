function [ yiqim ] = my_rgb2ntsc( rgbim )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    Y = 0.299*rgbim(:,:,1) +0.587*rgbim(:,:,2) +0.114*rgbim(:,:,3);
    I = 0.596*rgbim(:,:,1) -0.274*rgbim(:,:,2) -0.322*rgbim(:,:,3);
    Q = 0.211*rgbim(:,:,1) -0.523*rgbim(:,:,2) +0.312*rgbim(:,:,3);
    
    yiqim(:,:,1) = Y;
    yiqim(:,:,2) = I;
    yiqim(:,:,3) = Q;
    
end

