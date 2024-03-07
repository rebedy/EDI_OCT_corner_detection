# ---------------------------------------------------------------
# ### OCT Dataset Filing System ###
# > patients:30
# > oct images:1,439
# ---------------------------------------------------------------

import os
import cv2
from PIL import Image
from scipy import misc


# ### Patient folder and OCT image listing.
src_path = 'EDI_OCT_dataset_initial/'
data_dir = 'OCT_dataset/'

patient_folder = os.listdir(src_path)
patient_path = []
oct_imgs = []


def make_folder_path_ifnot_exists(src_path, folder_list, name='', a=None, b=None):
    folder_path = []
    for i, folder in enumerate(folder_list):
        folder_path.append(src_path + name + folder[a:b] + '/')
        if not os.path.exists(folder_path[i]):
            os.mkdir(folder_path[i])
    return folder_path


# ### Adjust to 2-digit number for original and marked OCT images
# * <b> 1. Original OCT images

def make_2digit_original(src_dir, img):
    if img[-6].isnumeric():
        pass
    else:
        new_oct = img[0:-5] + '0' + img[-5:]
        os.rename(src_dir + img, src_dir + new_oct)
        return new_oct


# * <b> 2. Marked OCT images

def make_2digit_marked(src_dir, img):
    if img[-7].isnumeric():
        pass
    else:
        new_oct = img[0:-6] + '0' + img[-6:]
        os.rename(src_dir + img, src_dir + new_oct)
        return new_oct


for i, patient in enumerate(patient_folder):
    patient_path.append(os.path.join(src_path + patient + '/'))
    oct_list = os.listdir(patient_path[i])

    for j, oct_list in enumerate(oct_list):
        if oct_list[-5] == 'c' or oct_list[-5] == 'C':
            make_2digit_marked(patient_path[i], oct_list)
        else:
            make_2digit_original(patient_path[i], oct_list)


# ### Sorting, cropping, fliping if 'Left' and saving in new directory as rename

def crop_and_flip(oct_path, oct_img, dir_path):
    image = Image.open(oct_path)
    crop_area = (168, 0, 680, 160)
    if patient[-1] == 'R':
        crop = image.crop(crop_area).save(dir_path + oct_img[0:-4] + '.R.png')
    if patient[-1] == 'L':
        crop = image.crop(crop_area).transpose(Image.FLIP_LEFT_RIGHT).save(dir_path + oct_img[0:-4] + '.L.png')
    return crop


original_path = []
marked_path = []

for i, patient in enumerate(patient_folder):
    oct_list = os.listdir(patient_path[i])
    oct_imgs.append(oct_list)
    original_path.append(data_dir + 'original_' + patient + '/')
    if not os.path.exists(original_path[i]):
        os.mkdir(original_path[i])
    marked_path.append(data_dir + 'marked_' + patient + '/')
    if not os.path.exists(marked_path[i]):
        os.mkdir(marked_path[i])

    # #### Crop and flip, and save as renaming by sorting into Marked or Original image

    for j in range(len(oct_imgs[i])):
        oct_path = os.path.join(src_path + patient + '/' + oct_imgs[i][j])

        if oct_imgs[i][j][-5] == 'c' or oct_imgs[i][j][-5] == 'C':
            crop_and_flip(oct_path, oct_imgs[i][j], marked_path[i])

        else:
            crop_and_flip(oct_path, str(oct_imgs[i][j]), original_path[i])


# ---------------------------------------------------------------
# ### Line Segmenting ###
# ---------------------------------------------------------------

def get_path(upper_path, list_name, slash=1):
    get_path = []
    if slash == 0:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name))

    if slash == 1:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name + '/'))

    return get_path


folder_list = os.listdir(data_dir)
marked_folder = []
for folder in folder_list:
    if folder[0] == 'm':
        marked_folder.append(folder)

marked_path = get_path(data_dir, marked_folder)
marked_oct = []


def new_dir_line_segmented_by_color(color_name):

    seg_path = 'line_segmented/'
    seg_folder_path = seg_path + color_name + 'segmented/'

    if not os.path.exists(seg_folder_path):
        os.mkdir(seg_folder_path)

    seg_line_path = []
    for i, folder in enumerate(marked_folder):
        seg_line_path.append(seg_folder_path + color_name + folder[7:] + '/')
        if not os.path.exists(seg_line_path[i]):
            os.mkdir(seg_line_path[i])
    return seg_line_path


def make_dir_by_color(upper_folder_list, path, color_name, tag):

    folder_path = path + color_name + tag

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    new_path = []
    for i, folder in enumerate(upper_folder_list):
        new_path.append(folder_path + color_name + folder[2:] + '/')
        if not os.path.exists(new_path[i]):
            os.mkdir(new_path[i])
    return new_path


line_path = new_dir_line_segmented_by_color('')
line_path = make_dir_by_color(marked_folder, 'line_segmented/', '1_', 'segmented/')

oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct = os.listdir(mpath)
    for moct in marked_oct:
        moct_path = os.path.join(mpath, moct)
        oct_path.append(moct_path)
        img = Image.open(moct_path)
        pixels = img.load()
        out = img.copy()
        for x in range(img.size[0]):
            for y in range(img.size[1]):  # red, yellow, green, blue
                if pixels[x, y] == (255, 255, 0):
                    out.putpixel((x, y), (225, 225, 225))
                else:
                    out.putpixel((x, y), 0)
        out.save(line_path[i] + moct)


line_path = new_dir_line_segmented_by_color('1_')
line_path = make_dir_by_color(marked_folder, 'line_segmented/', '1_', 'segmented/')

oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct = os.listdir(mpath)
    for moct in marked_oct:
        moct_path = os.path.join(mpath, moct)
        oct_path.append(moct_path)
        img = Image.open(moct_path)
        pixels = img.load()
        out = img.copy()
        for x in range(img.size[0]):
            for y in range(img.size[1]):  # red, yellow, green, blue
                if pixels[x, y] == (255, 255, 0):
                    out.putpixel((x, y), (225, 225, 225))
                else:
                    out.putpixel((x, y), 0)
        out.save(line_path[i] + moct)


# ---------------------------------------------------------------
# ### Extracting Line Pixel(MASK-GRAY) ###
# ---------------------------------------------------------------
line_path = new_dir_line_segmented_by_color('2_')

oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct = os.listdir(mpath)
    for moct in marked_oct:
        moct_path = os.path.join(mpath, moct)
        oct_path.append(moct_path)
        img = Image.open(moct_path)
        pixels = img.load()
        out = img.copy()
        for x in range(img.size[0]):
            for y in range(img.size[1]):  # red, yellow, green, blue
                if pixels[x, y] == (255, 0, 0) or pixels[x, y] == (255, 255, 0) or pixels[x, y] == (128, 255, 0) or pixels[x, y] == (0, 255, 255):
                    out.putpixel((x, y), (225, 225, 225))
                else:
                    out.putpixel((x, y), 0)
        out.save(line_path[i] + moct)


# ### Extracting Line Pixel(MASK-RGB) ###

line_path = new_dir_line_segmented_by_color('LineRGB_')

oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct = os.listdir(mpath)
    for moct in marked_oct:
        moct_path = os.path.join(mpath, moct)
        oct_path.append(moct_path)

        img = Image.open(moct_path)
        pixels = img.load()
        out = img.copy()
        for x in range(img.size[0]):
            for y in range(img.size[1]):  # red, yellow, green, blue
                if pixels[x, y] == (255, 0, 0) or pixels[x, y] == (255, 255, 0) or pixels[x, y] == (128, 255, 0) or pixels[x, y] == (0, 255, 255):
                    pass
                else:
                    out.putpixel((x, y), 0)
        out.save(line_path[i] + moct)


# ---------------------------------------------------------------
# ### Morphological Transformation
# * <b>  "1_segmented" folder morphology
#                 ->  Dilate + Blur + only yellow line in binary
# * <b>  "2_segmented" folder morphology
#                 ->  Dilate + Blur + all in binary
# * <b>  "3_segmented" folder morphology
#                 ->  Dilate + only yellow line in binary
# * <b>  "4_segmented" folder morphology
#                 ->  Dilate + all in binary
# ---------------------------------------------------------------

line_seg_path = 'line_segmented/'
seg_path = '1_segmented/'
seg_folder_path = os.path.join(line_seg_path, seg_path)
line_folder_list = os.listdir(seg_folder_path)
morph_path = 'morph_segmented/'
if not os.path.exists(morph_path):
    os.mkdir(morph_path)
morph_folder_path = os.path.join(morph_path + '1_segmented/')
if not os.path.exists(morph_folder_path):
    os.mkdir(morph_folder_path)


for i, line_folder in enumerate(line_folder_list):
    line_folder_path = os.path.join(seg_folder_path + line_folder + '/')
    line_list = os.listdir(line_folder_path)
    for img_name in line_list:
        img_path = os.path.join(line_folder_path, img_name)
        img = cv2.imread(img_path)    # shape (160,512,3)
        kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilate = cv2.dilate(img, kernel_d, iterations=2)
        blur = cv2.GaussianBlur(dilate, (9, 9), 0)

        misc.imsave(morph_folder_path + img_name, blur)
