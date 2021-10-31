"""
python generate_Masks.py
Takes Input JSONs from labelme Combined Classwise Masks and Instance Based Classwise Masks
Author : khanfarhan10

Instructions for Running:

pip install virtualenv
virtualenv LabelMeEnv
LabelMeEnv\Scripts\activate
conda.bat deactivate (optional)
pip install -r requirements.txt
python generate_Masks.py
"""


import fnmatch
import cv2
import json
import numpy as np
import os
import base64
from copy import deepcopy

import imageio
import numpy as np

import shutil
# import matplotlib.pyplot as plt

LABELLINGS = {"car": 1,"zebra crossing":2}
LABEL_MAPPINGS = {v: k for k, v in LABELLINGS.items()}
CLASSES = list(LABELLINGS.keys())

ROOT_DIR = os.getcwd()


"""
# DEFINE YOUR VARIABLES HERE
DATA_PATH :
change this to the directory inside which you have json files
it will find using regex, so it can be a parent folder inside
which there are subdirectories which finally contain jsons
(Basically doesn't care about folder structure, gets all JSONs inside DATA_PATH)

Output_Combined_Masks_Dir:
Output Directory where all the Combined Masks Files will be written
Files will be written in the format : FileNameClassName.png

Output_Single_Masks_Dir :
Output Directory where all the Single Mask Files will be written
Files will be written in the format : FileName.png

Output_Instance_Masks_Dir:
Output Directory where all the Instance (Single) Masks Files will be written
Files will be written in the format : FileNameClassNameInstanceNumber.png

Output_Images_Dir:
Output Directory where all the Images extracted from JSONs Files will be written
Files will be written in the format : FileName.png

save_semantic :
Save Semantic Masks (Combined)
save_instance :
Save Instance Masks (Single Masks)
save_one_mask :
Save to a single Semantic Mask
Note : The Generated Masks might look black but hold a lot of data!
For proper visualization of Masks, see Notebook : Visualize_Single_Masks.ipynb
"""
DATA_PATH = os.path.join(ROOT_DIR, "LabelMeData\images\zebra_crossing")
print(DATA_PATH)
Output_Combined_Masks_Dir = "Final_Output_Combined_Masks"
Output_Single_Masks_Dir = "Final_Output_Single_Masks"
Output_Instance_Masks_Dir = "Final_Output_Instance_Masks"
Output_Images_Dir = "Final_Output_Images"
save_semantic = True
save_instance = True
save_one_mask = True
"""
# END OF VARIABLES
"""

Masks_Dir = Output_Instance_Masks_Dir
Images_Dir = Output_Images_Dir
Combined_Masks_Dir = Output_Combined_Masks_Dir


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    else:
        print("Directory already existed : ", dir)
    return dir


def create_folders(paths):
    for epath in paths:
        create_dir(epath)


DIRS_TO_CREATE = [Masks_Dir, Images_Dir,
                  Combined_Masks_Dir, Output_Single_Masks_Dir]

create_folders(DIRS_TO_CREATE)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def openJSON(json_path):
    with open(json_path) as f:
        jsonfile = json.load(f)
    return jsonfile


def save_base64_to_img(Base64ImageData, OutputImgPath):
    with open(OutputImgPath, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(Base64ImageData))


def getFileNameNoExt(fpath):
    """
    get filenamee no extension
    """
    return os.path.splitext(os.path.basename(fpath))[0]


def saveCombinedClassWiseMasks(ShapesData, ShapeX, ShapeY, filename, saveone=False):
    # Create common masks for semantic segmentation - UNET
    CLASS_MASKS = dict()
    mask = np.zeros([ShapeX, ShapeY], dtype=np.uint8)
    filler_value = 255  # 255 or 1
    for each_class in CLASSES:
        CLASS_MASKS[each_class] = deepcopy(mask)
    for i in range(len(ShapesData)):
        current_label = ShapesData[i]["label"]
        filler_data = np.array(
            [ShapesData[i]["points"]], dtype=np.int32)
        print(filler_data.shape)
        CLASS_MASKS[current_label] = cv2.fillPoly(
            CLASS_MASKS[current_label], filler_data, filler_value)
    for each_class in CLASSES:
        CurrentMaskPath = os.path.join(Combined_Masks_Dir, str(
            filename)+str("_")+str(each_class)+".png")
        cv2.imwrite(CurrentMaskPath, CLASS_MASKS[each_class])

    if saveone == True:
        OneMaskPath = os.path.join(
            Output_Single_Masks_Dir, str(filename)+".png")
        onemask = np.zeros([ShapeX, ShapeY], dtype=np.uint8)
        for each_class in CLASSES:
            segmap = CLASS_MASKS[each_class]
            idx_X, idx_Y = segmap.nonzero()
            onemask[idx_X, idx_Y] = LABELLINGS[each_class]
        cv2.imwrite(OneMaskPath, onemask)
    return CLASS_MASKS


def saveInstanceClassWiseMasks(ShapesData, ShapeX, ShapeY, filename):
    # Create unique masks for instance segmentation - MRCNN
    counter = 0
    LABELS = []
    LABELS_DATA = dict()
    for each_val in ShapesData:
        mask = np.zeros((ShapeX, ShapeY))
        counter += 1
        label = each_val["label"]
        LABELS.append(label)
        MaskFile = str(filename)+str(label)+str(counter).zfill(4)+".png"
        MaskPath = os.path.join(Masks_Dir, MaskFile)
        points = each_val["points"]
        print(label)
        AREA_POINTS = []
        for [x, y] in points:
            print("(" + str(round(x, 2)) + "," + str(round(y, 2)) + ")", end="|")
            AREA_POINTS.append([x, y])
        print()

        # if label is already present in dictionary then append else add it
        if label in LABELS_DATA:
            # already present
            # [[x0, y0],[x1, y1],[x2, y2]]
            TEMP_AREA_POINTS = LABELS_DATA[label]
            # NEW_AREA_POINTS [[[x3, y3],[x4, y4],[x5, y5]],[[x0, y0],[x1, y1],[x2, y2]]]
            # = TEMP_AREA_POINTS [[x3, y3],[x4, y4],[x5, y5]] + (Current) AREA_POINTS [[x0, y0],[x1, y1],[x2, y2]]
            NEW_AREA_POINTS = TEMP_AREA_POINTS + [AREA_POINTS]
            LABELS_DATA[label] = NEW_AREA_POINTS
        else:
            # not present
            LABELS_DATA[label] = [AREA_POINTS]

        AREA_POINTS = np.array(AREA_POINTS)
        #print("Area Points : " + str(AREA_POINTS))
        cv2.fillPoly(mask,  np.int32([AREA_POINTS]), (255))
        print("Correct Ones :", np.int32([AREA_POINTS]))
        cv2.imwrite(MaskPath, mask)
    return LABELS, LABELS_DATA


if __name__ == '__main__':
    json_files = find('*.json', DATA_PATH)
    print(len(json_files), "JSON Files Found :")

    AnnotatedImgFolder = DATA_PATH

    for jsonpath in json_files:
        content = openJSON(jsonpath)

        filename = getFileNameNoExt(jsonpath)

        print("FileName :", filename)
        ImgData = content["imageData"]
        ImagePath = os.path.join(Images_Dir, str(filename)+".png")

        save_base64_to_img(Base64ImageData=ImgData, OutputImgPath=ImagePath)

        imgFile = cv2.imread(ImagePath)
        ShapeX, ShapeY, ShapeZ = imgFile.shape
        if save_semantic:
            CLASS_MASKS = saveCombinedClassWiseMasks(
                ShapesData=content["shapes"], ShapeX=ShapeX, ShapeY=ShapeY, filename=filename)
        if save_instance:
            LABELS, LABELS_DATA = saveInstanceClassWiseMasks(
                content["shapes"], ShapeX, ShapeY, filename)
        if save_one_mask:
            CLASS_MASKS = saveCombinedClassWiseMasks(
                ShapesData=content["shapes"], ShapeX=ShapeX, ShapeY=ShapeY, filename=filename, saveone=True)

    print("Generation Complete.")
