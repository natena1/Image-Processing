# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:38:53 2023

@author: AM4
ЛАБОРАТОРНАЯ РАБОТА №6"""

import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
from utility import segmentation_utils

image = cv.imread('../images/mandarin.jpg')
plt.imshow(image)
plt.show()

'''Уменьшаем размер изображения'''
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
image_2 = cv.resize(image, dim, interpolation = cv.INTER_AREA)
plt.imshow(image_2)
plt.show()

image_hsv = cv.cvtColor(image_2, cv.COLOR_RGB2HSV)
gray = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)
rgb_image = cv.cvtColor(image_2, cv.COLOR_BGR2RGB)


from mpl_toolkits.mplot3d import Axes3D
## Основанные на регионах (Region-based)
# Разрастание областей (region growing)
# определяем координаты начальных точек
seeds = [(340, 510), (280, 460), (200, 440), (275, 580), (240, 515), (280, 500), (215, 400)]
# координаты для графика
x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))
# порог похожести цвета региона
threshold = 93
# находим сегментацию? используя метод из segmentation_utils
segmented_region = segmentation_utils.region_growingHSV(image_2, seeds, threshold)
# накладываем маску - отображаем только участки попавшие в какой-либо сегмент
result = cv.bitwise_and(image_2, image_2, mask=segmented_region)
# отображаем полученное изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(image_2, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()


# Разделение областей
qt = segmentation_utils.QTree(stdThreshold = 0.25, minPixelSize = 5,img = image_2.copy()) 
qt.subdivide()
tree_image = qt.render_img(thickness=0, color=(0,0,0))

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image_2, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.show()

from mpl_toolkits.mplot3d import Axes3D
## Основанные на регионах (Region-based)
# Разрастание областей (region growing)
# определяем координаты начальных точек
seeds = [(340, 510), (280, 460), (200, 440), (275, 580), (240, 515), (280, 500), (215, 400)]
#seeds = [(340, 510), (280, 460), (200, 440), (275, 580), (240, 515)]
# координаты для графика
x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))
# порог похожести цвета региона
threshold = 93
# находим сегментацию используя метод из segmentation_utils
segmented_region = segmentation_utils.region_growingHSV(image_2, seeds, threshold)
# накладываем маску - отображаем только участки попавшие в какой-либо сегмент
result = cv.bitwise_and(tree_image, tree_image, mask=segmented_region)
# отображаем полученное изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()
