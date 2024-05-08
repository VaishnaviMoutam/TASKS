## ITERATE OF FIRST 10 NUMBERS

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
   
```pip install numpy ```
```pip install opencv-python```
```pip install matplotlib```
 
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

## Input:

![a](https://github.com/VaishnaviMoutam/TASKS/assets/169046827/02097b31-f76f-481e-9c54-f2a04041451f)

## Output:

![graph](https://github.com/VaishnaviMoutam/TASKS/assets/169046827/dff78259-e36e-4a85-a0bb-4d5253edf9c2)


## WEBCAM VIDEO CAPTURING

## What is Webcam
A webcam is a video camera which is designed to record or stream to a computer or computer network. They are primarily used in video telephony, live streaming and social media, and security.

## REQUIRED PACKAGES 
```opencv```

```pip install opencv-python```

## Example Program

import the opencv library 
     import cv2 
define a video capture object 
     vid = cv2.VideoCapture(0) 
  
     while(True): 
      
Capture the video frame 
by frame 

     ret, frame = vid.read() 
  
Display the resulting frame

     cv2.imshow('frame', frame) 
      
the 'q' button is set as the 
quitting button you may use any 
desired button of your choice 

     if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
After the loop release the cap object 

     vid.release()
     
Destroy all the windows 

     cv2.destroyAllWindows()

## Output:

[Screencast from 08-05-24 12:08:07 PM IST.webm](https://github.com/VaishnaviMoutam/TASKS/assets/169046827/571978f4-9165-4985-9ef9-85f014644587)


## CROPING & DRAWING BOUNDING BOXES

## Bounding Boxes:

A bounding box in essence, is a rectangle that surrounds an object, that specifies its position, class and confidence(how likely it is to be at that location). Bounding boxes are mainly used in the task of object detection, where the aim is identifying the position and type of multiple objects in the image.

## Required Packages:
```os, csv, PIL```

```pip install Pillow```

## Example Program:
1.Importing Libraries:

os: Provides functions for interacting with the operating system, such as creating directories and joining file paths.

csv: Allows reading and writing CSV files.
  
PIL.Image and PIL.ImageDraw: These modules from the Python Imaging Library (PIL) provide functions for image manipulation and drawing shapes on images.
  
     import os
     import csv
     from PIL import Image,ImageDraw

2.File Paths:

csv_file: Path to the CSV file containing bounding box information.
    
image_dir: Directory containing the images.
    
output_dir: Directory where the processed images with bounding boxes will be saved.
     
     csv_file = "/home/vaishnavi-moutam/Downloads/7622202030987_bounding_box(12).csv"
     image_dir = "/home/vaishnavi-moutam/Downloads/7622202030987(1)/7622202030987"
     output_dir = "/home/vaishnavi-moutam/Downloads/7622202030987(1)/7622202030987_with_boxes"
     os.makedirs(output_dir, exist_ok=True)

3.Function Definitions:

draw_boxes(image, boxes): Draws bounding boxes on the input image using PIL's ImageDraw module.
    
crop_image(image, boxes): Crops the input image based on the bounding box coordinates provided and returns a list of cropped images.

     def draw_boxes(image, boxes):
       draw = ImageDraw.Draw(image)
       for box in boxes:
          left = int(box['left'])
          top = int(box['top'])
          right = int(box['right'])
          bottom = int(box['bottom'])
          draw.rectangle([left, top, right, bottom], outline="blue")
      return image
     def crop_image(image, boxes):
       cropped_images = []
       for box in boxes:
         left = int(box['left'])
         top = int(box['top'])
         right = int(box['right'])
         bottom = int(box['bottom'])
         cropped_img = image.crop((left, top, right, bottom))
         cropped_images.append(cropped_img)
      return cropped_images

4.Main Processing:

The script opens the CSV file and iterates over each row using csv.DictReader. Each row represents bounding box information for an image.
    
For each row, it retrieves the image file name, constructs the full path to the image, and loads the image using PIL's Image.open.
    
It extracts the bounding box coordinates from the CSV row and creates a list of dictionaries containing box coordinates.
    
It then calls crop_image to crop the image based on the bounding box coordinates. Each cropped image is saved with a filename prefixed by an index.
    
It calls draw_boxes to draw bounding boxes on the full image and saves the result.


     with open(csv_file, 'r') as file:
       csv_reader = csv.DictReader(file)
        for row in csv_reader:
          image_name = row['filename']
          image_path = os.path.join(image_dir, image_name)
          output_path = os.path.join(output_dir, image_name)
          image = Image.open(image_path)
          boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
          cropped_images = crop_image(image, boxes)
          for i, cropped_img in enumerate(cropped_images):
             cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))

5.Output:

Cropped images with bounding boxes are saved in the output_dir, prefixed with an index to differentiate multiple boxes in the same image.
    
Full images with bounding boxes drawn on them are also saved in the output_dir.










    
