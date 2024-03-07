
# * #### Shi-Tomasi Corner Detector  ####
# # And getting coordinate and writing on CSV

import os
import numpy as np
import cv2

import pandas as pd


def get_path(upper_path, list_name, slash=1):
    get_path = []
    if slash == 0:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name))
    if slash == 1:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name + '/'))
    return get_path


def new_corner_detected_dir(color_name):
    color_corner_path = corner_detection_path + 'corner' + color_name + '_detected/'
    if not os.path.exists(color_corner_path):
        os.mkdir(color_corner_path)
    return color_corner_path


def make_folder_path_ifnot_exists(src_path, folder_list, name=''):  # , a= int(), b=None):
    folder_path = []
    for i, folder in enumerate(folder_list):
        folder_path.append(src_path + name + folder[7:] + '/')
        if not os.path.exists(folder_path[i]):
            os.mkdir(folder_path[i])
    return folder_path


line_dir = 'line_segmented/'
lineYE_dir = line_dir + 'lineYE_segmented/'
lineGR_dir = line_dir + 'lineGR_segmented/'
lineYE_folder = os.listdir(lineYE_dir)
lineGR_folder = os.listdir(lineGR_dir)  # ['lineGR_01105040.hurinsoon.R',
lineYE_path = get_path(lineYE_dir, lineYE_folder)  # ['line_segmented/lineYE_segmented/lineYE_01105040.hurinsoon.R/',
lineGR_path = get_path(lineGR_dir, lineGR_folder)

corner_detection_path = 'corner_detection/'
cornerYE_folder = new_corner_detected_dir('YE')
cornerGR_folder = new_corner_detected_dir('GR')
cornerYE_path = make_folder_path_ifnot_exists(cornerYE_folder, lineYE_folder, name='cornerYE_')
cornerGR_path = make_folder_path_ifnot_exists(cornerGR_folder, lineGR_folder, name='cornerGR_')

# To mark Corner detection result on Morphological transformed image
corner_morphYE_folder = new_corner_detected_dir('_morphYE')
corner_morphGR_folder = new_corner_detected_dir('_morphGR')
corner_morphYE_path = make_folder_path_ifnot_exists(corner_morphYE_folder, lineYE_folder, name='corner_morphYE_')
corner_morphGR_path = make_folder_path_ifnot_exists(corner_morphGR_folder, lineGR_folder, name='corner_morphGR_')


# *********************************************
# ## Getting Coordinates by colors ##
# *********************************************


# *********************************************
# > ###  YELLOW ###
# >> - Dilation : RECT, 2x2, iter=2
# > - GaussianBlur : 9x9, 0
# > - goodFeaturesToTrack : 4, 0.05, 20
# > - circle(img, center, radius, color) : 1, 255, -1
# > - x-Coordinate criteria : 200 / 280
# *********************************************

def shi_Tomasi_and_Gaussian_YE(color_line_path):
    corner_coords = []
    oct_name = []
    for p, line_path in enumerate(color_line_path):
        line_oct = os.listdir(line_path)
        oct_name.append(line_oct)
        oct_path = get_path(line_path, line_oct, slash=0)
        for i, oct_path in enumerate(oct_path):
            img = cv2.imread(oct_path)
            # out = img.copy()

            # ...Morphology
            kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilate = cv2.dilate(img, kernel_d, iterations=2)
            blur = cv2.GaussianBlur(dilate, (9, 9), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            # ...Corner Detection
            corners = cv2.goodFeaturesToTrack(gray, 4, 0.05, 20)
            # ...Suitable Coordinate Selection
            abs_corners1, abs_corners2, abs_corners_index1, abs_corners_index2 = [], [], [], []
            real_corners = []
            for x in corners:
                abs_corners1.append(abs(200 - x[0][0]))
            abs_corners_index1 = np.argsort(abs_corners1)
            real_corners.append(corners[abs_corners_index1[0]])
            for x in corners:
                abs_corners2.append(abs(280 - x[0][0]))
            abs_corners_index2 = np.argsort(abs_corners2)
            real_corners.append(corners[abs_corners_index2[0]])

            # ...To mark corners and write image.
            # real_corners = np.int0(real_corners)
            corner_coords.append(real_corners)
            for c in real_corners:
                x, y = c.ravel()
                # ...line_segmented Image
                cv2.circle(img, (x, y), 1, 255, -1)
                cv2.imwrite(cornerYE_path[p] + line_oct[i], img)
                # ...Morphologied Image
                cv2.circle(blur, (x, y), 1, 255, -1)
                cv2.imwrite(corner_morphYE_path[p] + line_oct[i], blur)
    return oct_name, corner_coords


# ### YE_coords
oct_name, YE_coords = shi_Tomasi_and_Gaussian_YE(lineYE_path)


# *********************************************
# > ### GREEN - GaussianBlur + shi-Tomasi ###
# >> - Dilation : RECT, 2x2, iter=2
# > - GaussianBlur : 9x9, 0
# > - goodFeaturesToTrack : 2, 0.05, 5
# > - circle(img, center, radius, color) : 2, 255, -1
# *********************************************

def shi_Tomasi_and_Gaussian_GR(color_line_path):
    corner_coords = []
    oct_name = []
    for p, line_path in enumerate(color_line_path):
        line_oct = os.listdir(line_path)
        oct_name.append(line_oct)
        oct_path = get_path(line_path, line_oct, slash=0)
        for i, oct_path in enumerate(oct_path):
            img = cv2.imread(oct_path)
            # out = img.copy()

            # ...Morphology
            kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilate = cv2.dilate(img, kernel_d, iterations=2)
            blur = cv2.GaussianBlur(dilate, (9, 9), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            # ...Corner Detection
            corners = cv2.goodFeaturesToTrack(gray, 2, 0.05, 5)
            # ... correct error due to no corner value
            if corners is None:
                corner_coords.append(np.array([[[0, 0]], [[0, 0]]]))
                cv2.imwrite(cornerGR_path[p] + line_oct[i], img)
            # ... Sort by x-Coordinate
            else:
                # corners = np.int0(corners)
                if corners[0][0][0] > corners[1][0][0]:
                    sort = np.sort(corners.view('i8,i8'), order=['f0'], axis=0)  # .view(np.int0)
                    # corners.view('i8,i8').sort(order=['f0'],axis=0)
                    corner_coords.append(sort)
                else:

                    # ...Mark corners and write image
                    for c in corners:
                        x, y = c.ravel()
                    # ...Line_segmented Image
                        cv2.circle(img, (x, y), 2, 255, -1)  # (img, center, radius, color)
                        cv2.imwrite(cornerGR_path[p] + line_oct[i], img)
                    # ...Morphologied Image
                        cv2.circle(blur, (x, y), 2, 255, -1)   # (img, center, radius, color)
                        cv2.imwrite(corner_morphGR_path[p] + line_oct[i], blur)

    return oct_name, corner_coords


# ### GR_coords
oct_name, GR_coords = shi_Tomasi_and_Gaussian_GR(lineGR_path)


# *********************************************
# ## Setting Columns and Values for DataFrame
# *********************************************


# * ###  OCT slice name List

lineYE_folder[0]
lineYE_folder[0][7:15]
lineYE_folder[6][-1]
oct_name[0][0][0:-6]
slice_name = []
for i in range(len(lineYE_folder)):
    for j in range(len(oct_name[i])):
        slice_name.append(oct_name[i][j][0:-6])


# * ### patientID, patientName, R/L List

patient_id = []
patient_name = []
RorL = []
for i, folder in enumerate(lineYE_folder):
    for a in range(24):
        patient_id.append(lineYE_folder[i][7:15])     # 30 patients
        patient_name.append(lineYE_folder[i][16:-2])  # 30 patients
        RorL.append(lineYE_folder[i][-1])             # 30 patients


# * ### Coordinate by colors for Column

def get_coords_col_list(color):
    coords_col = []
    coords_col.append(color + '1_x')
    coords_col.append(color + '1_y')
    coords_col.append(color + '2_x')
    coords_col.append(color + '2_y')
    return coords_col


yellow_col = get_coords_col_list('YE')
green_col = get_coords_col_list('GR')


# *********************************************
# ## Creating DataFrame
# *********************************************


# ### 1. Filling in Patient's ID, Name, R/L, EDI OCT slice no. cells.
# ... Creating New DataFrame

patientINFO_col = ['patientID', 'patientName', 'RorL', 'oct slice no.']
df = pd.DataFrame(columns=patientINFO_col)
# ... OCT slice name & index
df.loc[:, 'oct slice no.'] = pd.Series(slice_name)
# ... patient infor. filling Rows
for loc in df.index:
    df.loc[df.index == loc, 'patientID'] = pd.Series(patient_id)
    df.loc[df.index == loc, 'patientName'] = pd.Series(patient_name)
    df.loc[df.index == loc, 'RorL'] = pd.Series(RorL)


# ### 2.  Filling coordinate value and append as new Column
# ** ... Appending columns on initial DataFrame **

def filling_coords_into_new_col(color_col, corner_coords):
    x1, y1, x2, y2 = [], [], [], []
    for i in range(len(corner_coords)):
        x1.append(corner_coords[i][0][0][0])
        y1.append(corner_coords[i][0][0][1])
        x2.append(corner_coords[i][1][0][0])
        y2.append(corner_coords[i][1][0][1])
    df[color_col[0]] = pd.Series(x1)
    df[color_col[1]] = pd.Series(y1)
    df[color_col[2]] = pd.Series(x2)
    df[color_col[3]] = pd.Series(y2)
    return x1, y1, x2, y2


YE1_x, YE1_y, YE2_x, YE2_y = filling_coords_into_new_col(yellow_col, YE_coords)
GR1_x, GR1_y, GR2_x, GR2_y = filling_coords_into_new_col(green_col, GR_coords)


# ### 3. Save as xlsx file.
df.to_excel('EDI_OCT_coord.xlsx', sheet_name='Sheet1', index=False)
