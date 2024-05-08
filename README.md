## PRINTING OF FIRST 10 NUMBERS

## This is the code explanation about printing of first 10 numbers and it performs the following tasks:

    num = list(range(10))
 It creates a list named 'num' containing numbers from 0 to 9 using the 'range()' function.

    previousNum = 0
 It initializes a variable 'previousNum' with a value of 0.

    for i in num:
It iterates through each element 'i' in the 'num' list.

    sum = previousNum + i
Inside the loop, it calculates the sum of the current number 'i' and the previous number 'previousNum'.
    
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))
It prints the current number 'i', the previous number 'previousNum', and their sum 'sum' as a formatted string.
    
    previousNum=i   
It updates the value of 'previousNum' to the current number 'i' for the next iteration.

## After all this we get output:

Current Number 0Previous Number 0is 0

Current Number 1Previous Number 0is 1

Current Number 2Previous Number 1is 3

Current Number 3Previous Number 2is 5

Current Number 4Previous Number 3is 7

Current Number 5Previous Number 4is 9

Current Number 6Previous Number 5is 11

Current Number 7Previous Number 6is 13

Current Number 8Previous Number 7is 15

Current Number 9Previous Number 8is 17


##  HISTOGRAM OF AN IMAGE

## WHAT IS HISTOGRAM OF AN IMAGE:

A histogram of an image represents the distribution of pixel intensity values across the image. In a grayscale image, the intensity values typically range from 0 (black) to 255 (white), and the histogram shows how many pixels have each intensity value.

## ADVANTAGES OF HISTOGRAM:

  1.It is a simple and effective technique that can be used to improve the contrast of images.
  
  2.It is fast and easy to implement.
  
  3.It can be used to improve the visibility of details in images with low contrast.
  
  4.It is a versatile technique that can be used in a variety of applications.

## REQUIRED PACKAGES
   numpy, opencv, matplotlib
   
```pip install numpy opencv matplotlib```
 
## Code Explanation about Histogram of an image

    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt

numpy: This library provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.

cv2: This is the OpenCV library used for image processing tasks.

matplotlib.pyplot: This submodule of Matplotlib provides a MATLAB-like plotting framework.
 
    img = cv.imread('/home/vaishnavi-moutam/Desktop/v/B.jpg')
    cv.imwrite("/home/vaishnavi-moutam/Desktop/v/a.jpg",img)
   
cv.imread: This function reads an image from the specified file path (/home/vaishnavi-moutam/Desktop/v/B.jpg) into a NumPy array. The image is loaded in BGR (Blue, Green, Red) format.
cv.imwrite: This function writes the loaded image (img) to another file path (/home/vaishnavi-moutam/Desktop/v/a.jpg). It seems like this line is unnecessary and can be removed.

     assert img is not None, "file could not be read, check with os.path.exists()"
This line checks if the image was successfully loaded. If img is None, it raises an assertion error with the message "file could not be read, check with os.path.exists()". However, it seems like there's no need for this assertion because if the image loading fails, cv.imread will return None and the code will already terminate due to the assertion error.

     color = ('b','g','r')
     for i,col in enumerate(color):
      histr = cv.calcHist([img],[i],None,[256],[0,256])
      plt.plot(histr,color = col)
      plt.xlim([0,256])
     plt.show()
    
color: This tuple contains the colors for plotting the histograms. Here, ('b', 'g', 'r') represents blue, green, and red channels, respectively.

The for loop iterates over each channel (blue, green, and red) using enumerate.

Inside the loop, cv.calcHist calculates the histogram for each channel (i) of the image (img). It computes the histogram with 256 bins (intensity levels) ranging from 0 to 255.

plt.plot is used to plot the histogram.

plt.xlim([0, 256]) sets the limits of the x-axis from 0 to 255.

Finally, plt.show() displays the plotted histograms.





    
