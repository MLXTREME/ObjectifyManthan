# Objectify - Team CodeDevs (Manthan 2021)
Object Detection, Semantic and Instance Segmentation - 
By *Team CodeDevs* for Competition *Manthan21*
Code will be released prior Idea Submission Selection.

<center>
<a href="#"><img width="50%" height="auto"  src="https://user-images.githubusercontent.com/78292851/139589354-64f6d6cd-6af8-47f1-aba7-3f87cc92c3a3.png" height="80px"/></a>
</center>
    
- Vehicles (Truck, Bus, Boat, Airplane)
- Roads Signage - (Zebra Crossing, Traffic Light)
- Man-Made Architectures - (Buildings, Bridges)

### DataSets
- COCO - The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.
- CityScapes - Cityscapes is a large-scale database which focuses on semantic understanding of urban street scenes. It provides semantic, instance-wise, and dense pixel annotations for 30 classes grouped into 8 categories (flat surfaces, humans, vehicles, constructions, objects, nature, sky, and void). Data was captured in 50 cities during several months, daytimes, and good weather conditions with over 25k images.

### Objectives :
1. Fully Working Web application : Allowing the user to Upload Image.
2. Perform Instance Segmentation plus Object Detection - Creating Annotations over the Uploaded Image with Bounding Boxes and Class Names, and Pixel Labelling.
3. Displaying the Output Image with the Annotated Object on it.

### Our Idea :
1. Our web app takes an input image from the user using JavaScript
2. The respective image gets saved in the locally hosted centralised  SQL database.
3. The model will fetch the object and will detect the same, using the libraries tensorflow, pytorch & pixellib with Deep Learning Models such as PointRend and MobileNetV3. 
4. The input image gets annotated using cv2 libraries.
5. The annotated objects that has been detected, gets displayed along with the original uploaded image via the Django Backend.
6. The model gives results in the form of a JSON (JavaScript Object Notation) format and the output is displayed with CSS and HTML website on the local web server.

### TechStack Used :
- Python
- JavaScript
- Django
- Tensorflow
- Pytorch

## Tech Stack:

<p align="left">
    <a href="https://html.com/" target="_blank"> <img src="https://img.icons8.com/nolan/64/html.png"/> </a>
    <a href="https://www.tensorflow.org/" target="_blank"> <img src="https://img.icons8.com/external-justicon-lineal-color-justicon/64/000000/external-css-file-file-type-justicon-lineal-color-justicon.png"/> </a>
    <a href="https://www.tensorflow.org/" target="_blank"> <img src="https://img.icons8.com/color/48/000000/javascript--v1.png"/> </a>
    <a href="https://www.tensorflow.org/" target="_blank"> <img src="https://img.icons8.com/color/48/000000/tensorflow.png"/> </a>
    <a href="https://www.python.org" target="_blank"> <img src="https://img.icons8.com/color/48/000000/python.png"/> </a> 
    <a style="padding-right:8px;" href="https://www.djangoproject.com/" target="_blank"> <img src="https://img.icons8.com/ios-filled/50/4a90e2/django.png"/> </a>
   <a href="https://git-scm.com/" target="_blank"> <img src="https://img.icons8.com/color/48/000000/git.png"/> </a> 
     <a href="https://opencv.org/" target="_blank"> <img src="https://img.icons8.com/color/48/000000/opencv.png"/>></a>

</p>



# Object Detection using Pointrend Model:

![maskboxes](https://user-images.githubusercontent.com/78292851/139590557-26d106c5-b080-40b0-837e-caf03a7015f0.jpg)

For performing segmentation of the objects in images and videos, PixelLib library is used, and so we have invoked the same in our respective project. 
PixelLib provides support for Pytorch and it uses PointRend for performing more accurate and real time instance segmentation of objects in images and videos. Hence, annotations over the image takes place once the work is done.


# Instance Segmentation using MobileNetV3:


![unnamed](https://user-images.githubusercontent.com/78292851/139590680-7c86e652-411a-41bf-b047-c6e4ebf0311b.png)
![unnamed (1)](https://user-images.githubusercontent.com/78292851/139590682-b8fed9ac-74a0-45dd-b282-8762673b37db.png)

The implementation of the MobileNetV3 architecture follows closely the original paper and it is customizable and offers different configurations for building Classification, Object Detection and Semantic Segmentation backbones. 
Furthermore, it was designed to follow a similar structure to MobileNetV2 and the two share common building blocks. The MobileNetV3 class is responsible for building a network out of the provided configuration. The models are then adapted and applied to the tasks of object detection and semantic segmentation. 
For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder to achieve new state of the art results for mobile classification, detection and segmentation. Finally, the project tries to faithfully implement MobileNetV3 for real-time semantic segmentation, with the aims of being efficient, easy to use, and extensible.

## Instance Segmentation vs. Object Detection :



![object-detection](https://user-images.githubusercontent.com/78292851/139591596-7d5eef8f-2fb3-4819-b7f1-d62fcfd3bb4b.gif)



![1_J33mxWAtCSEV1GsWV3vKLQ](https://user-images.githubusercontent.com/78292851/139591258-600d7341-66e0-47d9-85c7-409eaf4d01c3.gif)
