# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:37:50 2023

@author: Acer
"""

'''# Лабораторная работа 3. Цифровая обработка изображений.'''

import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

# Изменим стандартный размер графиков matplotlib
plt.rcParams["figure.figsize"] = [6, 4]

'''# 1. Метрики качества. Среднеквадратическая ошибка (MSE). Пиковое отношение сигнал/шум (PSNR).

Метрики качества служат для измерения схожести/различия между двумя изображениями.'''

def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2)  #|I1 - I2|
    s1 = np.float32(s1)  # cannot make a square on 8 bits
    s1 = s1 * s1  # |I1 - I2|^2
    sse = s1.sum()  # sum elements per channel
    if sse <= 1e-10:  # sum channels
        return 0  # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

'''## 1.2 SSIM'''

def getSSIM(i1, i2):
    C1 = 6.5025  # only for 8-bit images
    C2 = 58.5225  # only for 8-bit images
    # INITS
    I1 = np.float32(i1)  # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2  # I2^2
    I1_2 = I1 * I1  # I1^2
    I1_I2 = I1 * I2  # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv.divide(t3, t1)  # ssim_map =  t3./t1;
    ssim = cv.mean(ssim_map)  # mssim = average of ssim map
    ssim = ssim[:3]
    return ssim

"""1. Добавим зашумленное изображение:"""

image1 = cv.imread('../images/zadanie3.jpg')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

im1 = rgb_image1 

"""## 2 Нелинейная фильтрация полутоновых изображений"""
median_image1 = cv.medianBlur(im1, 3)
median_image2 = cv.medianBlur(im1, 5)

gs = plt.GridSpec(1, 3)
plt.figure(figsize=(8, 10))
plt.subplot(gs[0])
plt.imshow(im1, cmap='gray')
plt.title('Исходное изображение')

plt.subplot(gs[1])
plt.imshow(median_image1, cmap='gray')
psnr = util.getPSNR(im1, median_image1)
ssim = util.getSSIM(im1, median_image1)
plt.title(f'Восстановленное изображение \n '
          f'медианным фильтром 3х3 \n'
          f'PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')

plt.subplot(gs[2])
plt.imshow(median_image2, cmap='gray')
psnr = util.getPSNR(im1, median_image2)
ssim = util.getSSIM(im1, median_image2)
plt.title(f'Восстановленное изображение \n '
          f'медианным фильтром 5х5 \n'
          f'PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')

plt.show()

"""
# 3. Линейная фильтрация изображений в пространственной области.
**Гауссиан.**
"""
kernel55 = np.ones((5, 5), np.float32) / 25
kernel77 = np.ones((7, 7), np.float32) / 49

filtered_image1 = cv.filter2D(im1, -1, kernel55)
filtered_image2 = cv.filter2D(im1, -1, kernel77)
gaussian_image1 = cv.GaussianBlur(im1, (3, 3), 0)
gaussian_image2 = cv.GaussianBlur(im1, (5, 5), 0)

# вывод
gs = plt.GridSpec(2, 4)
plt.figure(figsize=(15, 12))

plt.subplot(gs[0, 0])
plt.xticks([]), plt.yticks([])
plt.title('Исходное зашумленное изображение')
plt.imshow(im1, cmap='gray')

plt.subplot(gs[1, 0])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат средней линейной \n фильрации с ядром 5х5 \n '
          f'PSNR = {util.getPSNR(im1, filtered_image1):.3f} \n '
          f'SSIM = {util.getSSIM(im1, filtered_image1):.3f}')
plt.imshow(filtered_image1, 'gray')

plt.subplot(gs[1, 1])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат средней линейной \n фильрации с ядром 7х7 \n '
          f'PSNR = {util.getPSNR(im1, filtered_image2):.3f} \n '
          f'SSIM = {util.getSSIM(im1, filtered_image2):.3f}')
plt.imshow(filtered_image2, 'gray')

plt.subplot(gs[1, 2])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат гауссовской  \n фильрации с ядром 3х3 \n '
          f'PSNR = {util.getPSNR(im1, gaussian_image1):.3f} \n '
          f'SSIM = {util.getSSIM(im1, gaussian_image1):.3f}')
plt.imshow(gaussian_image1, 'gray')

plt.subplot(gs[1, 3])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат гауссовской  \n фильрации с ядром 5х5 \n '
          f'PSNR = {util.getPSNR(im1, gaussian_image2):.3f} \n '
          f'SSIM = {util.getSSIM(im1, gaussian_image2):.3f}')
plt.imshow(gaussian_image2, 'gray')

plt.show()

