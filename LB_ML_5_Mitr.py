# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:26:45 2023

@author: Acer
"""

'''# Лабораторная работа 5. Цветовые модели. Цветовая сегментация'''
# Commented out IPython magic to ensure Python compatibility.

import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

"""Загружаем изображение. Преобразуем в модель RGB"""

image = cv.imread('../images/mandarin.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()

'''Уменьшаем размер изображения'''
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
image_rgb2 = cv.resize(image_rgb, dim, interpolation = cv.INTER_AREA)
plt.imshow(image_rgb2)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

"""Отображаем разные каналы по разным осям на трехмерном графике. В случае модели RGB не видно кластеризации по цвету."""
r, g, b = cv.split(image_rgb2)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = image_rgb2.reshape((np.shape(image_rgb2)[0]*np.shape(image_rgb2)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()


"""Преобразуем изображение в цветовую модель HSV"""

image_hsv = cv.cvtColor(image_rgb2, cv.COLOR_RGB2HSV)

"""Отобразим разные каналы полученного изображения на трехмерном графике"""

h, s, v = cv.split(image_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()


"""2. С использованием OpenCV можно визуализировать гистограммы по каналам."""

histSize = [256]
range = [0, 256]


def plot_rgb_hist(image, histSize, range):
    histSize = [256]
    range = [0, 256]
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv.calcHist([image], [i], None, histSize, range)
        plt.plot(hist, color=col)
        plt.xlim(range)


plot_rgb_hist(image_rgb2, histSize, range)
plt.show()

"""Создадим маски для разных оттенков оранжевого цвета"""

lower_orange = np.array([4,200,100])
upper_orange = np.array([25,255,255])
lo_square = np.full((10, 10, 3), lower_orange, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), upper_orange, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lo_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(do_square))
plt.show()

"""Найдем на изображении цвета подходящие под маску И добавим маску на изображение."""

mask = cv.inRange(image_hsv, lower_orange, upper_orange)
result = cv.bitwise_and(image_rgb2, image_rgb2, mask=mask)

plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb2)
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(result)
plt.show()

"""Зададим другую маску"""

light_white = (0, 60, 200)
dark_white = (15, 255, 255)

lw_square = np.full((10, 10, 3), light_white, dtype=np.uint8) / 255.0
dw_square = np.full((10, 10, 3), dark_white, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lw_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(dw_square))
plt.show()

mask_white = cv.inRange(image_hsv, light_white, dark_white)
result_white = cv.bitwise_and(image_hsv, image_hsv, mask=mask_white)
plt.figure(figsize=(15,20))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.subplot(1, 3, 2)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(result_white)
plt.show()

"""Применим обе маски и сгладим изображение"""

final_mask = mask + mask_white

final_result = cv.bitwise_and(image_rgb2, image_rgb2, mask=final_mask)
blur = cv.GaussianBlur(final_result, (7, 7), 0)

plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(final_result)
plt.subplot(1, 3, 3)
plt.imshow(blur)
plt.show()

"""Пробуем сделать цветовую сегментацию"""

def segment_image(image_rgb2):
    ''' Attempts to segment the whale out of the provided image '''

    # Convert the image into HSV
    hsv_image = cv.cvtColor(image_rgb2, cv.COLOR_RGB2HSV)

    # Set the blue range
    lower_blue = (4, 200, 100)
    upper_blue = (25, 255, 255)

    # Apply the blue mask
    mask = cv.inRange(hsv_image, lower_blue, upper_blue)

    # Set a white range
    light_white = (0, 60, 200)
    dark_white = (15, 255, 255)

    # Apply the white mask
    mask_white = cv.inRange(hsv_image, light_white, dark_white)

    # Combine the two masks
    final_mask = mask + mask_white
    result = cv.bitwise_and(image_rgb2, image_rgb2, mask=final_mask)

    # Clean up the segmentation using a blur
    blur = cv.GaussianBlur(result, (7, 7), 0)
    return blur


result = segment_image(image_rgb2)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb2)
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

"""Самостоятельное задание. Сделать цветовую сегментацию выданного изображения.
Пороги для цветовой сегментации подобрать в ручную, используя преобразование изображения
в формат LAB.

### Еще пример цветовой сегментации

Известно, что цветные цифровые изображения представляют собой совокупность трех 
цветовых плоскостей, каждая из которых характеризует одну независимую 
составляющую цвета, представленную в том же формате, что и обычное 8-битное 
полутоновое изображение. Следовательно, все описанные процедуры обработки 
полутоновых изображений в яркостной области могут быть обобщены и на случай 
обработки цветных изображений. Специфика же здесь связана прежде всего с 
различными цветовыми моделями, позволяющими по-разному работать с разными 
цветовыми и другими составляющими изображения.
"""

h, s, v = cv.split(image_hsv)

low_h = 4
high_h = 25

mask = cv.inRange(h, low_h, high_h)
result = cv.bitwise_and(image_rgb2, image_rgb2, mask=mask)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(image_rgb2)
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.imshow(mask, cmap='gray')
plt.title('Маска')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[2])
plt.hist(h.reshape(-1), np.max(h), [np.min(h), np.max(h)])
plt.vlines(low_h, 0, 5000, 'r'), plt.vlines(high_h, 0, 5000, 'r')
plt.title('Гистограмма h слоя')
plt.subplot(gs[3])
plt.imshow(result)
plt.title('Изображение с пикселями выделенного цвета')
plt.show()



