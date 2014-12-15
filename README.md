Colorization using Optimization
============

Colorization is a process of making black-and-white images colorful. A novel method for colorization ([Levin, Lischinski & Weiss, 2004](http://www.cs.huji.ac.il/~yweiss/Colorization/colorization-siggraph04.pdf)) has been proposed that uses the simple idea that neighboring pixels in space-time have similar intensities. Although the authors provide a MATLAB implementation of their method, we all love Python! Thus, our project aims to write these code in Python.

University of Tartu Advanced Algorithmics project (Fall 2014)

============
USAGE

Use your favorite image editing program
(e.g. Photoshop, gimp) and save two images to
the disk. 

1) The original B/W image. The image needs to be saved in an RGB (3 channels) format.
2) The B/W with colors scribbled in the desired places. Use your favorite paint program (Photoshop, paint, gimp and each) to generate the scribbles. Make sure no compression is used and the only pixels in which the RGB value of the scribbled image are different then the original image are the colored pixels.

NB! For scribbling use exact tools such as Pencil in Photoshop and save it into a no compression format such as Bitmap (.bmp)

============
ORIGINAL ARTICLE READ ME

This package contains an implementation of the image colorization approach described in the paper:
A. Levin D. Lischinski and Y. Weiss Colorization using Optimization.
ACM Transactions on Graphics, Aug 2004. 
 

Usage of this code is free for research purposes only. 
Please refer to the above publication if you use the program.

Copyrights: The Hebrew University of Jerusalem, 2004.
All rights reserved.

Written by Anat Levin.
Please address comments/suggestions/bugs to: alevin@cs.huji.ac.il
