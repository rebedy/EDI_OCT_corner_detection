# * ### Corner Detection with Morphological Transformation ###

import os
import numpy as np
import cv2
from PIL import Image

import scipy
from scipy import ndimage
from skimage import img_as_float
from skimage.restoration import denoise_bilateral
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import matplotlib.pyplot as plt


def get_path(upper_path, list_name, slash=1):
    get_path = []
    if slash == 0:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name))

    if slash == 1:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name + '/'))

    return get_path


# Corner Detection with Morphological Transformation Example

lineYE_dir = 'line_segmented/lineYE_segmented/'
lineYE_folder = os.listdir(lineYE_dir)

lineYE_path = get_path(lineYE_dir, lineYE_folder)

corner_detection_path = 'trial_corner_detection/'
corner_folder_path = []

for i, folder in enumerate(lineYE_folder):
    corner_folder_path.append(corner_detection_path + 'cornerYE_' + folder[7:] + '/')
    if not os.path.exists(corner_folder_path[i]):
        os.mkdir(corner_folder_path[i])


# * ### Basic Form for listing images


for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)
        # or img = Image.open(ye_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# * ### ≪ Morphological Transformation ≫ ###
# > * ### Scikit-Image / skimage - Mathematical Morphology
# >> ** 1. Erosion and Dilation **
# >> ** 2. Denoising **

# #### [skimage] Erosion

for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = Image.open(ye_path)
        img_erod = ndimage.binary_erosion(img)
        scipy.misc.imsave(corner_folder_path[p] + line_oct[i], img_erod)


# ####  [skimage] Dilation


for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = Image.open(ye_path)
        img_dila = ndimage.morphology.binary_dilation(img)
        scipy.misc.imsave(corner_folder_path[p] + line_oct[i], img_dila)


# ####  [skimage] Dilation & Erosion (closing in cv2)


for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = Image.open(ye_path)
        img_dila = ndimage.morphology.binary_dilation(img)
        img_erod = ndimage.binary_erosion(img_dila)
        scipy.misc.imsave(corner_folder_path[p] + line_oct[i], img_erod)


# ####  [skimage] Denoising


for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = Image.open(ye_path)
        noisy = img_as_float(img)

        # img3 = denoise_bilateral(noisy, sigma_range=0.1, sigma_spatial=15, multichannel=True)
        # # `sigma_range` has been deprecated in favor of `sigma_color`. The `sigma_range` keyword argument will be removed
        denoised = denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15, multichannel=True)
        scipy.misc.imsave(corner_folder_path[p] + line_oct[i], denoised)


# > * ### OpenCV - Morphological Transformation
# >> ** 1. Erosion & Dilation and Opening & Closing **
# > ** 2. Image Smoothing - Filtering(LPF) **
# > ** 3. Image Blurring - Blur, Gussian Blur, Median Blur, Bilateral **


# ####  [OpenCV] Erosion and Dilation


# Kernel : MORPH_RECT:사각/ MORPH_ELLIPSE:타원형/ MORPH_CROSS:십자 모양
for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)  # shape (160,512,3)
        # or img = Image.open(ye_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # shape (160,512)

        # Erosion
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        erode = cv2.erode(img, kernel_e, iterations=1)

        # Dilation
        kernel_d = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilate = cv2.dilate(img, kernel_d, iterations=1)
        dilate_gray = cv2.cvtColor(dilate, cv2.COLOR_BGR2GRAY)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow('image', img)
        cv2.imshow('erode', erode)
        cv2.imshow('dilate', dilate)
        # cv2.imwrite(corner_detection_path +'dilated(cross,3x3,1)_'+line_oct[i], dilate)
        # cv2.imwrite(corner_folder_path[p] +'dilated_'+line_oct[i], dilate)
        cv2.imwrite(corner_folder_path[p] + line_oct[i], dilate_gray)


# ####  [OpenCV]  Opening  ( Erosion ☞ Dilation ) &  Closing  ( Dilation ☞Erosion )

# Kernel : MORPH_RECT:사각/ MORPH_ELLIPSE:타원형/ MORPH_CROSS:십자 모양
for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)

        # Opening
        kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_o, iterations=10)

        # Closing
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_c, iterations=1)

        cv2.imshow('img', img)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', opening)
        cv2.waitKey(0)


# ####  [OpenCV] Image Smoothing - Filtering - LPF : Low-pass Filter, 고주파제거

for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)

        def nothing(x):
            pass

        cv2.namedWindow('image')
        cv2.createTrackbar('Kernel', 'image', 1, 20, nothing)

        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
            k = cv2.getTrackbarPos('Kernel', 'image')

            # (0,0)이면 에러가 발생함으로 1로 치환
            if k == 0:
                k = 1

            # trackbar에 의해서 (1,1) ~ (20,20) kernel생성
            kernel = np.ones((k, k), np.float32) / (k * 2)
            dst = cv2.filter2D(img, -1, kernel)

            scipy.misc.imsave(corner_folder_path[p] + 'filter_' + line_oct[i], denoised)


# ####  [OpenCV] Image Blurring

for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)

        # pyplot를 사용하기 위해서 BGR을 RGB로 변환.
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        # 일반 Blur
        dst1 = cv2.blur(img, (3, 3))

        # GaussianBlur
        dst2 = cv2.GaussianBlur(img, (9, 9), 0)

        # Median Blur
        dst3 = cv2.medianBlur(img, 3)

        # Bilateral Filtering
        dst4 = cv2.bilateralFilter(img, 4, 10, 10)  # 9,75,75

        images = [img, dst1, dst2, dst3, dst4]
        titles = ['Original', 'Blur(3X3)', 'Gaussian Blur(9X9)', 'Median Blur', 'Bilateral']

        for i in range(5):
            plt.subplot(3, 2, i + 1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.imsave(corner_folder_path[p] + titles[i] + '_' + line_oct[i], images[i])


# ### * ≪ Corner Detection ≫ * ###
#    > * ### Scikit-Image / skimage
# >> ** 1. corner_harris + corner_subpix: **  -> slow, but good
# > ** 2. corner_kitchen_rosenfeld:**
# > ** 3. corner_shi_tomasi:**
# > ** 4. corner_foerstner: **
# > ** 5. corner_moravec: **
# > ** 6. corner_fast: **
# > ** 7. corner_orientations: **


# #### [ skimage ]  corner_harris + corner_subpix

# skimage - corner_harris + corner_subpix
for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # skimage
        coords = corner_peaks(corner_harris(gray), min_distance=10)
        coords_subpix = corner_subpix(gray, coords, window_size=15)

        fig, ax = plt.subplots()
        ax.imshow(gray, interpolation='nearest', cmap=plt.cm.gray)
        ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
        ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=5)
        ax.axis((0, 512, 168, 0))
        plt.axis('off')

        # save the figure to file
        fig.savefig(corner_folder_path[p] + line_oct[i], bbox_inches='tight', pad_inches=0, frameon=False)
        plt.close(fig)


# ####  [ skimage ] corner_shi_tomasi

for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# > ### * OpenCV * ###
# >> ** 1. Harris Corner Detector: ** 빠르기 중간, pointer is too bold
# > ** 2. Harris Corner with SubPixel Accuracy: ** 아주빠름, pointer is too small
# > ** 3. Shi-Tomasi Corner Detector: **

# OpenCV - Harris Corner Detector in OpenCV
for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)
        out = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # openCV (grayscale&float32, 이웃pixelrange, Sobel미분 인자, Harris 검출수학식 R의 k값)
        dst = cv2.cornerHarris(gray, 2, 1, 0.0001)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.0005 * dst.max()] = [0, 0, 255]  # mark the corners with red dots

        # cv2.imwrite(corner_folder_path[p] +line_oct[i],img)

        dst = np.uint8(dst)
        plt.imshow(img)
        plt.imshow(dst, alpha=0.05)
        plt.show()


# #### [OpenCV] Harris Corner Detector with SubPixel Accuracy

# OpenCV - Corner with SubPixel Accuracy
for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)
        out = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # openCV - find Harris corners
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 15, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (20, 20), (-1, -1), criteria)

        # Now draw them
        res = np.hstack((centroids, corners))
        img[res[:, 1], res[:, 0]] = [0, 0, 255]
        img[res[:, 3], res[:, 2]] = [0, 255, 0]

        cv2.imwrite(corner_folder_path[p] + line_oct[i], img)


# ####  [OpenCV] Shi-Tomasi Corner Detector

# YELLOW : OpenCV - Shi-Tomasi Corner Detector
corner_coords = []
for p, path in enumerate(lineYE_path):
    line_oct = os.listdir(path)
    ye_path = get_path(path, line_oct, slash=0)

    for i, ye_path in enumerate(ye_path):
        img = cv2.imread(ye_path)
        out = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # openCV (grayscale image, # of corners, 코너검출품질-코너로 판단할 문턱값, 최소거리)
        corners = cv2.goodFeaturesToTrack(gray, 4, 0.001, 10)
        if corners is None:
            corner_coords.append(corners)
            cv2.imwrite(corner_folder_path[p] + line_oct[i], img)
            pass
        else:
            # corners = np.int0(corners)
            corner_coords.append(corners)
            for c in corners:
                x, y = c.ravel()
                cv2.circle(img, (x, y), 2, 255, -1)   # (img, center, radius, color)
                cv2.imwrite(corner_folder_path[p] + line_oct[i], img)


# GREEN : OpenCV - Shi-Tomasi Corner Detector
corner_coords = []
for p, path in enumerate(lineGR_path):
    line_oct = os.listdir(path)
    gr_path = get_path(path, line_oct, slash=0)

    for i, gr_path in enumerate(gr_path):
        img = cv2.imread(gr_path)
        out = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # openCV
        corners = cv2.goodFeaturesToTrack(gray, 4, 0.001, 10)
        if corners is None:
            corner_coords.append(corners)
            cv2.imwrite(corner_folder_path[p] + line_oct[i], img)
            pass
        else:
            # corners = np.int0(corners)
            corner_coords.append(corners)
            for c in corners:
                x, y = c.ravel()
                cv2.circle(img, (x, y), 2, 255, -1)   # (img, center, radius, color)
                cv2.imwrite(corner_folder_path[p] + line_oct[i], img)


# ### ≪ Corner Detection after Morphological Transformation and Getting Coordinates ≫ ###


# * #### Corner Detection after morphological transformation Example ####


# setting img for trial
ex_img = cv2.imread('Gaussian Blur(5X5).png')
gray = cv2.cvtColor(ex_img, cv2.COLOR_BGR2GRAY)


# Harris
coords = corner_peaks(corner_harris(gray), min_distance=10)
coords_subpix = corner_subpix(gray, coords, window_size=10)

fig, ax = plt.subplots()
ax.imshow(gray, interpolation='nearest', cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=5)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=3)
ax.axis((0, 512, 168, 0))

plt.show()
fig.savefig('corner_filtering.png')


# Shi-Tomasi
corners = cv2.goodFeaturesToTrack(gray, 4, 0.001, 100)
# corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(ex_img, (x, y), 2, 255, -1)  # (img, center, radius, color)

plt.imshow(ex_img)
plt.show()
