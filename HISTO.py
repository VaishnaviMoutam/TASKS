import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 
import argparse

a = argparse.ArgumentParser()
a.add_argument("input",help="enter path")
a.add_argument("output",help="output path")
b = a.parse_args()
input=cv.imread(b.input)

 
#img = cv.imread('/home/vaishnavi-moutam/Desktop/v/B.jpg')
#cv.imwrite("/home/vaishnavi-moutam/Desktop/v/a.jpg",img)
assert input is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([input],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()
