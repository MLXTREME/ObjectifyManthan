from django.shortcuts import render,redirect
from django.http.response import HttpResponse
from .forms import UploadForm
from .models import FormModel,SaveModel
from django.contrib import messages
from django.http import JsonResponse

from django.core.files import File

import json
import numpy as np
import os
import cv2

global ins
IS_ML = True
MODEL_PATH =  "G:\GitLab\Objectify\LabelMeData\pointrend_resnet50.pkl" #  "D:\GITHUB MLXTREME\ML\LabelMeData\pointrend_resnet50.pkl"
global selection_type
selection_type = 0

"""
0 : Object Detection : PointRend
1 : Instance Detection (Map) : MobileV3Large 
2 : Instance Detection (Blend) : MobileV3Large 
"""

CURRENT_FOLDER = os.getcwd() # os.path.dirname(__file__)
print("Current Directory :",CURRENT_FOLDER)

def saveImg(filename, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, img)
    return True


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return super(NpEncoder, self).default(obj)

def writeJSON(newdict, json_path = 'static/result.json'):
    x = json.dumps(newdict, cls=NpEncoder,indent=4)
    # print(x)
    with open(json_path, 'w') as fp:
        fp.write(x)
    return True

def runPointrendModel(picture_path = os.path.join(os.path.dirname(__file__),  "images", "art.jpg"), 
                        output_image_name="media/images/output11.png"):
    r , output = ins.segmentImage(picture_path, show_bboxes=True, output_image_name=output_image_name)
    return r, output

#[top left x position, top left y position, width, height].
def createDict(r):
    newdict={}
    for i,(bbox, cname, sc) in enumerate(zip(r["boxes"] , r["class_names"], np.array(r["scores"]))): 
        newdict[i+1]=cname,bbox,sc
    return newdict

def simpleDict(r):
    cnames = []
    for cname in r["class_names"]:
        cnames.append(cname)
    cnames = list(set(cnames))
    return [{"label" : cnames}]

if IS_ML:
    import pixellib
    from pixellib.torchbackend.instance import instanceSegmentation

    ins = instanceSegmentation()
    ins.load_model(MODEL_PATH)


def appindex(request):
     return render(request,"index.html")

def test(request):
    context = {}
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data.get("Image")
            obj =FormModel.objects.create(img = img,)
            obj.save()
            print(obj)
            return redirect('result')    
    else:
        form = UploadForm()
    context['form']= form
    return render( request, "upload.html", context)  

def predict(request):
    if request.method =='GET':
        images = FormModel.objects.last() # all()
        # print("Image Type :", type(images.__class__.__name__))
        # print("Direc :", dir(images))
        print("Image Path :", images.img)
        print("IDs :", images.id)
        image_path = os.path.join(CURRENT_FOLDER, "media", str(images.img))
        image_id = images.id
        if IS_ML:
            r, output = runPointrendModel(image_path)
            newdict = createDict(r)
            writeJSON(simpleDict(r), os.path.join(CURRENT_FOLDER,"WebApp", "static","demo.json"))
            # images = os.path.join(CURRENT_FOLDER, "media", "images",  "out.png") 
            # G:\GitLab\Manthan21\WebApp\static\images
            images = os.path.join(CURRENT_FOLDER, "WebApp", "static", "images", "out.png") 
            print("Saving Image to : ",images)
            saveImg(filename = images, img = output)
            images = os.path.join("media", "images",  "out.png") 
        else:
            # for image in images:
            #     new_images = image.img.url
            # images = new_images
            print("Image Type :", type(images.__class__.__name__))
            print("Direc :", dir(images))
            images = [images]
        #     {% for image in images %}
        # <img src="{{image.img.url}}" />
        # {%endfor%}
        # "{%url_for('static', filename='od.png' )%}"
        print("Output Image Path :", images)
        return render(request, 'predict.html',{'images' : images})    
    elif(request.method == 'POST'):
        if request.POST.get('cars') and request.POST.get('cars1') :
            data=SaveModel()
            data.modeltype =request.POST.get('cars')
            data.task=request.POST.get('cars1')
            data.save()
            img = FormModel.objects.all()
            image = np.array(img)
            print(image.shape)
            #   print("Image Type :", type(img.__class__.__name__))
            #   print("Direc :", dir(img))
            return render(request, 'result.html')
    return render(request, 'index.html')

def result(request):
    return render(request, "result.html")

# G:\GitLab\Manthan21\media\images\out.png