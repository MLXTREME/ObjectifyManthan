# Objectify

View the [Google Docs Here](https://docs.google.com/document/d/1TTxFOF6SWhxWqJ88dU-PhzMHPpHtwRJc4lvUJLsL8cs/edit)

Object Detection - by Team CodeDevs for Competition Manthan

# Installing using a VirtualEnv

### Initially

pip install virtualenv (if you dont have it installed already)

virtualenv ObjEnv  
ObjEnv\Scripts\activate

pip install -r requirements.txt
pip list -> ObjEnv_full_reqs.txt

ipython kernel install --user --name=ObjEnv  
Installed kernelspec ObjEnv in C:\Users\farha\AppData\Roaming\jupyter\kernels\objenv

### Once Installed
 
ObjEnv\Scripts\activate  
conda.bat deactivate   

pip list

### Installing and Downloading Models

cd LabelMeData
python initial_install.py
python pointrend_inference.py

### Running the application

cd ActualApp 
python app.py

<!-- !pip install --upgrade mxnet
!pip install fastseg -->
