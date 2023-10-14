# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 22:11:31 2023

@author: Acer
"""

"""Подключение библиотек:"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utility import util

def lut(image):
    gray_image1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist1 = cv.calcHist([gray_image1], [0], None, [256], [0, 256]) # ф-ия считает гистораммы
    sh = np.sum(hist1)
    im_ek2 = np.ones(gray_image1.shape, dtype = np.uint8)
    for i in range(gray_image1.shape[0]):
        for j in range(gray_image1.shape[1]):
            sum_hist1 = np.cumsum(hist1)
            im_ek2[i,j] = ( 255 * (sum_hist1[gray_image1[i,j]]/sh))            
    return im_ek2


image = cv.imread('lenna.png')
cv.imshow("Image", image)
cv.waitKey(0)
cv.destroyAllWindows()

ek_image = lut(image)

cv.imshow("Image", ek_image)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("Image", gray_image1)
cv.waitKey(0)
cv.destroyAllWindows()
