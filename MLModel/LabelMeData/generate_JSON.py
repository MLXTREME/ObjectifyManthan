"""
python generate_JSON.py
"""


import fnmatch
import cv2
import json
import numpy as np
import os
import base64
from copy import deepcopy
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import shutil
# import matplotlib.pyplot as plt
import imantics
try:
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage
except:
    from imgaug.augmentables.segmaps import SegmentationMapOnImage as SegmentationMapsOnImage

ia.seed(1)

LABELLINGS = {"car": 1,"zebra crossing":2}
CLASSES = list(LABELLINGS.keys())

ROOT_DIR = os.getcwd()



augmentation = iaa.Sometimes(0.5, [
    iaa.CLAHE(clip_limit=(1, 10)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])

# Define our augmentation pipeline.
seq1 = iaa.Sequential([
    iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    # rotate by -45 to 45 degrees (affects segmaps)
    iaa.Affine(rotate=(-45, 45)),
    # apply water effect (affects segmaps)
    iaa.ElasticTransformation(alpha=50, sigma=5)
], random_order=True)


# Define our augmentation pipeline.
seq = iaa.Sequential([
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    # rotate by -45 to 45 degrees (affects segmaps)
    iaa.Affine(rotate=(-45, 45)),
    iaa.CLAHE(clip_limit=(1, 10))
], random_order=True)

"""
# DEFINE YOUR VARIABLES HERE
DATA_PATH :
change this to the directory inside which you have json files
it will find using regex, so it can be a parent folder inside 
which there are subdirectories which finally contain jsons
(Basically doesn't care about folder structure, gets all JSONs inside DATA_PATH)

num_augs: Number of Augmentations you need to perform

Final_Augmentation_Sequence : 
Choose an imgaug Augmentation Pipeline
Experiment with many pipelines and see what works best for you buddy
I have provided 3 sample augmentation pipelines

OutputJSONDir: 
Output Directory where all the JSON Files will be written
When you are generating using various augmentation techniques, you
It probably makes more sense to rename them as
Final_Output_JSONs_Imgs_Aug_Blur
Final_Output_JSONs_Imgs_Aug_SomeOtherName etc
You getting me?
"""
DATA_PATH = os.path.join(ROOT_DIR, "LabelMeData\images\zebra_crossing")
num_augs = 3
Final_Augmentation_Sequence = seq
OutputJSONDir = "Final_Output_JSONs_Imgs"

"""
# END OF VARIABLES
"""

Masks_Dir = "Temp_Masks"
Images_Dir = "Temp_Images"
Combined_Masks_Dir = "Temp_Combined_Masks"
AugMasksDir = "Temp_Aug_Masks"
# AugImgsDir = "Temp_Aug_Imgs"

TEMPORARY = [Masks_Dir, Images_Dir,
             Combined_Masks_Dir, AugMasksDir]

OutputDir = OutputJSONDir
AugImgsDir = OutputJSONDir

def destruct_folders(paths):
    for epath in paths:
        try:
            shutil.rmtree(epath)
        except:
            pass

destruct_folders(TEMPORARY)
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
                  Combined_Masks_Dir, AugMasksDir, AugImgsDir, OutputJSONDir]

create_folders(DIRS_TO_CREATE )
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


def saveCombinedClassWiseMasks(ShapesData, ShapeX, ShapeY, filename):
    CLASS_MASKS = dict()
    mask = np.zeros([ShapeX, ShapeY], dtype=np.uint8)
    for each_class in CLASSES:
        CLASS_MASKS[each_class] = deepcopy(mask)
    for i in range(len(ShapesData)):
        current_label = ShapesData[i]["label"]
        filler_data = np.array(
            [ShapesData[i]["points"]], dtype=np.int32)
        print(filler_data.shape)
        CLASS_MASKS[current_label] = cv2.fillPoly(
            CLASS_MASKS[current_label], filler_data, 255)  # 255 or 1
    for each_class in CLASSES:
        CurrentMaskPath = os.path.join(Combined_Masks_Dir, str(
            filename)+str("_")+str(each_class)+".png")
        cv2.imwrite(CurrentMaskPath, CLASS_MASKS[each_class])
    return True


def getCombinedMaskstoOneMask(filename, Combined_Masks_Dir, ShapeX, ShapeY):
    """
    Usage :
    big_segmap, ALL_SEGMAPS = getCombinedMaskstoOneMask(filename, Combined_Masks_Dir, ShapeX, ShapeY)
    """
    ALL_SEGMAPS = dict()
    # mapped segmapconstruct here
    big_segmap = np.zeros([ShapeX, ShapeY], dtype=np.uint8)
    for each_class in CLASSES:
        CurrentMaskPath = os.path.join(Combined_Masks_Dir, str(
            filename)+str("_")+str(each_class)+".png")
        old_segmap = np.array(imageio.imread(CurrentMaskPath), dtype=bool)
        print("Segmap", old_segmap.shape)
        ALL_SEGMAPS[each_class] = old_segmap
        label_num = LABELLINGS[each_class]
        idx_R, idx_C = np.nonzero(old_segmap)
        for r, c in zip(idx_R, idx_C):
            big_segmap[r][c] = label_num
    return big_segmap, ALL_SEGMAPS


def augmentSingle(image, big_segmap, augmentation_sequence):
    """
    Usage :
    augmented_image , augmented_segmap_mask = augmentSingle(image, big_segmap, augmentation_sequence=seq)
    """
    segmap = SegmentationMapsOnImage(big_segmap, shape=image.shape)
    # Augment images and segmaps.
    images_aug_i, segmaps_aug_i = augmentation_sequence(
        image=image, segmentation_maps=segmap)
    return images_aug_i, segmaps_aug_i.get_arr()


def breakOneMasktoCombinedMasks(augmented_segmap_mask, ShapeX, ShapeY):
    """
    Usage :
    AUG_MASKS = breakOneMasktoCombinedMasks(augmented_segmap_mask, ShapeX, ShapeY)
    """
    AUG_MASKS = dict()
    for each_class in CLASSES:
        label_num = LABELLINGS[each_class]
        idx_R, idx_C = np.where(augmented_segmap_mask == label_num)
        curr_segmap = np.zeros([ShapeX, ShapeY], dtype=bool)
        for r, c in zip(idx_R, idx_C):
            curr_segmap[r][c] = True
        AUG_MASKS[each_class] = curr_segmap
    return AUG_MASKS


def writeAugmentedMasksandImage(AUG_MASKS,augmented_image, AugMasksDir,AugImgsDir,filename,counter):
    """
    Usage :
    writeAugmentedMasksandImage(AUG_MASKS,augmented_image, AugMasksDir,AugImgsDir,filename,counter)
    """
    AugImgPath = os.path.join(AugImgsDir, str(
        filename)+str("_")+str(counter)+".png")

    cv2.imwrite(AugImgPath, augmented_image)
    for each_class in CLASSES:
        AugMaskPath = os.path.join(AugMasksDir, str(
            filename)+str("_")+str(each_class)+str(counter)+".png")
        curr_aug_mask = AUG_MASKS[each_class]
        
        cv2.imwrite(AugMaskPath, curr_aug_mask*255)
    return True        
        
def getShapesforJSONfromAugmentedMasks(AUG_MASKS,ShapeX, ShapeY):
    """
    Usage:
    all_shapes,NEW_MASKS = getShapesforJSONfromAugmentedMasks(AUG_MASKS,ShapeX, ShapeY)
    """
    NEW_MASKS = dict()
    all_shapes = []

    for each_class in CLASSES:  
        CLASS_MASKS = imantics.Polygons.from_mask(
            mask=AUG_MASKS[each_class])
        CLASS_MASKS_LIST = list(CLASS_MASKS)

        # 1, len(pts),2
        # np.zeros(segmap.shape)
        mask_final = np.zeros((ShapeX, ShapeY))

        # cv2.fillPoly( mask_final, BIG_AREA, 255) # 255 or 1
        BIG_AREA = []
        
        HUGE_AREA = np.zeros(
            (len(CLASS_MASKS_LIST), 1, len(CLASS_MASKS_LIST)//2, 2))
        big_counter = 0
        for each_small_list in CLASS_MASKS_LIST:
            Xs = each_small_list[0::2]
            Ys = each_small_list[1::2]
            """
            {'flags': {},
            'group_id': None,
            'label': 'building',
            'points': [[0.9525240599620731, 150.9941335024576],
                [0.0, 239.2364039955605]],
            'shape_type': 'polygon'}
            """
            coords = [[x, y] for x, y in zip(Xs, Ys)]
            dict_to_add = {'flags': {},
                            'group_id': None,
                            'label': each_class,
                            'points': coords,
                            'shape_type': 'polygon'}
            all_shapes.append(dict_to_add)
            # print(len(Xs),len(Ys))
            mask = np.zeros((1, len(Xs), 2))
            count = 0
            for x, y in zip(Xs, Ys):
                # print(count)

                mask[0][count] = np.array([x, y])
                count += 1
            # HUGE_AREA[big_counter] = mask
            big_counter += 1
            # print(mask)
            print(mask.shape)
            # mask_final = cv2.fillPoly( mask_final, [mask], 255) # 255 or 1
            mask_final = cv2.fillPoly(mask_final, np.array(
                [mask], dtype=np.int32), 255)  # 255 or 1
            BIG_AREA.append(mask)
        NEW_MASKS[each_class] = mask_final
        CLASS_MASKS_ARR = np.array(CLASS_MASKS_LIST,dtype=object)
    return all_shapes,NEW_MASKS


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

        CLASS_MASKS = saveCombinedClassWiseMasks(
            ShapesData=content["shapes"], ShapeX=ShapeX, ShapeY=ShapeY, filename=filename)

        

        image = imageio.imread(ImagePath)
        

        """
        Major Changes Here
        for each in Temp Combined - for a imagename , maskname label1,label2,label3 -> convert to aug_iamgename, maskname label1,label2,label3
        same behaviour required for SegmentationMapsOnImage
        """
        """
        imageData : base64
        imageHeight
        imagePath
        imageWidth
        shapes
        - label : building,
        - points : [[x1,y1],[x2,y2] ... [xn,yn]]
        """
        for counter in range(num_augs):
            big_segmap, ALL_SEGMAPS = getCombinedMaskstoOneMask(filename, Combined_Masks_Dir, ShapeX, ShapeY)
            augmented_image , augmented_segmap_mask = augmentSingle(image, big_segmap, augmentation_sequence=Final_Augmentation_Sequence)
            AUG_MASKS = breakOneMasktoCombinedMasks(augmented_segmap_mask, ShapeX, ShapeY)
            writeAugmentedMasksandImage(AUG_MASKS,augmented_image, AugMasksDir,AugImgsDir,filename,counter)
            all_shapes,NEW_MASKS = getShapesforJSONfromAugmentedMasks(AUG_MASKS,ShapeX, ShapeY)
            # OutputJSONDir
            OutputJSONPath = os.path.join(OutputJSONDir, str(
                filename)+str("_")+str(counter)+".json")
            # shutil.copy(jsonpath, OutputJSONPath)
            contentnew = openJSON(jsonpath)
            CounterImgPath = os.path.join(AugImgsDir, str(
                filename)+str("_")+str(counter)+".png")
            fh = CounterImgPath
            image_opened = open(fh, 'rb')
            image_read = image_opened.read()
            # encodestring also works aswell as decodestring
            image_64_encode = base64.encodebytes(image_read)

            # im
            # str(image_64_encode)
            contentnew["imageData"] = image_64_encode.decode()

            contentnew["imageHeight"] = ShapeX
            contentnew["imageWidth"] = ShapeY

            """
            '..\\all_annt_jpgData_palmdel_512\\PauliRGB (6)_palmdell.jpg'
            """
            add_path_before = "..\\all_annt_jpgData_palmdel_512\\"
            # AugImgsDir
            contentnew["imagePath"] = CounterImgPath
            
            contentnew["shapes"] = all_shapes
            print("Content Type :", type(contentnew))

            def np_encoder(object):
                if isinstance(object, np.generic):
                    return object.item()
            with open(OutputJSONPath, "w") as write_file:
                json.dump(contentnew, write_file, default=np_encoder)
    # shutil.copy(AugImgsDir,OutputJSONDir)
    print("Generation Complete.")
    destruct_folders(TEMPORARY)
