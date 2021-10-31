# Objectify - Team CodeDevs (Manthan 2021)
Object Detection, Semantic and Instance Segmentation - by Team CodeDevs for Competition Manthan
Code will be released prior Idea Submission Selection.

### Objects of Interest
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

