"""
ObjEnv\Scripts\activate
cd LabelMeData
python pointrend_inference.py
"""
import json
import numpy as np
import pixellib, os
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")

picture_path = os.path.join(os.path.dirname(__file__),  "images", "art.jpg")
r,output=ins.segmentImage(picture_path, show_bboxes=True, output_image_name="output11.png")

#[top left x position, top left y position, width, height].

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return super(NpEncoder, self).default(obj)

newdict={}

for i,(bbox, cname, sc) in enumerate(zip(r["boxes"] , r["class_names"], np.array(r["scores"]))): 
    newdict[i+1]=cname,bbox,sc

x = json.dumps(newdict, cls=NpEncoder,indent=4)
print(x)
with open('result.json', 'w') as fp:
    fp.write(x)
