"""
cd Objectify
ObjEnv\Scripts\activate
cd MXNet
python run_inference_mobile_v3_large.py
"""

from fastseg import MobileV3Large
from fastseg.image import colorize, blend

import cv2

import torch

if torch.cuda.is_available():
  model = MobileV3Large.from_pretrained().cuda().eval()
else:
  model = MobileV3Large.from_pretrained().eval()
 

def predict(imgpath):
    # read the image , convert BGR2RGB , perform mxnet prediction return mask result 
    imgvar=cv2.imread(imgpath)
    image = cv2.cvtColor(imgvar, cv2.COLOR_BGR2RGB )
    labels = model.predict_one(image)
    colorized = colorize(labels)
    return colorized

predict('car.jpg')

import torch
print(torch.__version__)