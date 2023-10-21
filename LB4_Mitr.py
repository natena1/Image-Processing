# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:18:27 2023

@author: Acer
"""

# Лабораторная работа 4. Цифровая обработка изображений.

'''В этой работе рассматриваются различные виды цифровой обработки изображений.

Цель лабораторной работы:
1. Бинаризация
2. Выделение границ
'''

# Commented out IPython magic to ensure Python compatibility.
import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

# Изменим стандартный размер графиков matplotlib
plt.rcParams["figure.figsize"] = [6, 4]

'''## 1 Бинаризация'''

my_image1 = cv.imread('../images/zadanie.jpg')
gray_my_image = cv.cvtColor(my_image1, cv.COLOR_BGR2GRAY)


channels = [0]
histSize = [256]
range = [0, 256]

gs = plt.GridSpec(2, 1)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_my_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.hist(gray_my_image.reshape(-1), 256, range)
plt.show()

### 1.1 Бинаризация  (пороговая фильтрация).
threshold = 130
image = gray_my_image


ret, thresh1 = cv.threshold(image, threshold+5, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(image, threshold+10, 255, cv.THRESH_BINARY)
ret, thresh3 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
ret, thresh4 = cv.threshold(image, threshold-5, 255, cv.THRESH_BINARY)
ret, thresh5 = cv.threshold(image, threshold-10, 255, cv.THRESH_BINARY)

images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
plt.figure(figsize=(15, 8))
for i in np.arange(len(images)):
    plt.subplot(6, 1, i + 1)
    plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

cv.imshow('', images[4])

