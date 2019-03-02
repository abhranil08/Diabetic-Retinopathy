# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 03:51:43 2019

@author: Abhranil
"""



import skimage
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

#im=Image.fromarray

def preProcess(path):
    ##STANDARDIZE
    img=cv2.imread(path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    downsample = skimage.measure.block_reduce(grayscale, (2,2), np.max)
    standardize = (downsample - downsample.mean()) / np.sqrt(downsample.var() + 1e-5)
    
    ##CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(grayscale)
    
    ##GAMMA CORRECTION
    adjusted=adjust_gamma(cl1,gamma=1.2)
    cv2.imwrite(path,adjusted)

def adjust_gamma(image, gamma=1.2):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


preProcess("14212_right.jpeg")

"""print(Image.fromarray(grayscale))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()"""