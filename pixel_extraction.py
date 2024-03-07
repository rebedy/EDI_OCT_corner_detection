# ---------------------------------------------------------------
# * ### Line Segmentation ###
# > original_folder : 30
# > original_oct : 719
# > marked_folder : 30
# > marked_oct : 720
# ---------------------------------------------------------------

import os
from PIL import Image


def get_path(upper_path, list_name, slash=1):
    get_path = []
    if slash == 0:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name))

    if slash == 1:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name + '/'))
    return get_path


data_dir = 'OCT_dataset/'

folder_list = os.listdir(data_dir)

marked_folder = []
for folder in folder_list:
    if folder[0] == 'm':
        marked_folder.append(folder)

marked_path = get_path(data_dir, marked_folder)

marked_oct = []


# ---------------------------------------------------------------
# * ### Make new directory for segmented images by color ###
# > #### Modulization - mkdir new segmented line folder
# ---------------------------------------------------------------

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


lineYE_path = new_dir_line_segmented_by_color('lineYE_')
lineGR_path = new_dir_line_segmented_by_color('lineGR_')


# ---------------------------------------------------------------
# * ### Extracting pixels by colors and save ###
# ---------------------------------------------------------------


# ### 1. YELLOW

oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct = os.listdir(mpath)
    for moct in marked_oct:
        oct_path = os.path.join(mpath, moct)
        img = Image.open(oct_path)

        pixels = img.load()
        out = img.copy()

        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if pixels[x, y] == (255, 255, 0):
                    pass
                else:
                    out.putpixel((x, y), 0)
        out.save(lineGR_path[i] + moct)


# ### 2. GREEN

oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct = os.listdir(mpath)
    for moct in marked_oct:
        oct_path = os.path.join(mpath, moct)
        img = Image.open(oct_path)

        pixels = img.load()
        out = img.copy()

        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if pixels[x, y] == (128, 255, 0):
                    pass
                else:
                    out.putpixel((x, y), 0)
        out.save(lineGR_path[i] + moct)


'''
# RGB value of each colors

Yellow: if pixels[x,y] == (255,255,0):
        if pixels[x,y][0] >= 230 and pixels[x,y][1] >= 230 and pixels[x,y][2] <= 100:

Green:  if pixels[x,y] == (128,255,0):
        if pixels[x,y][0] <=  and pixels[x,y][1] >=  and pixels[x,y][2] <= :

Red:    if pixels[x,y] == (255,0,0):
        if pixels[x,y][0] >=  and pixels[x,y][1] >=  and pixels[x,y][2] <= :

Blue:   if pixels[x,y] == (0, 255,255):
        if pixels[x,y][0] >=  and pixels[x,y][1] >=  and pixels[x,y][2] <= :
'''


def extracting_pixels_by_color(line_path, R, G, B):

    oct_path = []
    for i, mpath in enumerate(marked_path):
        marked_oct = os.listdir(mpath)
        for moct in marked_oct:
            oct_path = os.path.join(mpath, moct)
            img = Image.open(oct_path)

            pixels = img.load()
            out = Image.new('RGB', img.size, (0, 0, 0))
            # black = (0, 0, 0)

            for x in range(img.size[0]):
                for y in range(img.size[1]):
                    if pixels[x, y] == (R, G, B):
                        out.putpixel((x, y), pixels[x, y])
                    else:
                        pass

            out.save(line_path[i] + moct)


extracting_pixels_by_color(lineYE_path, 225, 225, 0)


# ------------------------------------------------------------------------------------------
# * ###  Yellow + Green Line ###
# ------------------------------------------------------------------------------------------

# * #### New directory for extracted image


line_path = 'line_segmented/'
lineYG_path = []

for i, folder in enumerate(marked_folder):
    lineYG_path.append(line_path + 'lineYG_' + folder[7:] + '/')
    if not os.path.exists(lineYG_path[i]):
        os.mkdir(lineYG_path[i])


def make_folder_path_ifnot_exists(src_path, folder_list, name='', a=None, b=None):
    folder_path = []
    for i, folder in enumerate(folder_list):
        folder_path.append(src_path + name + folder[a:b] + '/')
        if not os.path.exists(folder_path[i]):
            os.mkdir(folder_path[i])

    return folder_path


line_path = 'line_segmented/'
lineYG_path = make_folder_path_ifnot_exists(line_path, marked_folder, 'lineYG_', 7, None)


# *  #### Extracting pixels by colors and save

oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct = os.listdir(mpath)
    for moct in marked_oct:
        oct_path = os.path.join(mpath, moct)
        img = Image.open(oct_path)

        pixels = img.load()
        out = img.copy()
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if pixels[x, y] == (255, 255, 0) or pixels[x, y] == (128, 255, 0):
                    pass
                else:
                    out.putpixel((x, y), 0)

        out.save(lineYG_path[i] + moct)


# ----------------------------------------------------------------------


# ### Many different way to extract pixels


for moct in oct_path:
    img = Image.open(moct)
    pixels = img.load()
    out = img.copy()

    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if pixels[x, y] == (255, 255, 0):
                pass
            if pixels[x, y] == (128, 255, 0):
                pass
            else:
                out.putpixel((x, y), 0)

    for i, folder in enumerate(marked_folder):
        for j in range(24):
            out.save(lineYG_path[i] + str(marked_oct[i][j]))

# ----------------------------------------------------------------------
oct_path = []
for i, mpath in enumerate(marked_path):
    marked_oct.append(os.listdir(mpath))
    for marked in marked_oct:
        oct_path.append(os.path.join(mpath, marked[i]))

img_list = []
for moct in oct_path:
    img_list.append(Image.open(moct))

for a, img in enumerate(img_list):
    pixels = img.load()
    out = img.copy()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if pixels[x, y] == (255, 255, 0):
                pass
            if pixels[x, y] == (128, 255, 0):
                pass
            else:
                out.putpixel((x, y), 0)

    for i, folder in enumerate(marked_folder):
        for j in range(24):
            out.save(lineYG_path[i] + marked_oct[i][j])


for i, path in enumerate(marked_path):
    for j, moct in enumerate(marked_oct):
        oct_path = os.path.join(path + moct)
        img = Image.open(oct_path)
        pixels = img.load()
        out = img.copy()

        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if pixels[x, y] == (255, 255, 0) or pixels[x, y] == (128, 255, 0):
                    pass
                else:
                    out.putpixel((x, y), 0)

        out.save(lineYG_path[i] + moct)
