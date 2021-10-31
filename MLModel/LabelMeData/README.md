# Recreate JSON Augment LabelMe

Recreate the JSON from LabelMe using Augmentation from Imgaug.  
Helps in better Augmented Image Visualization via GUI.  
TODO : Add VGG IMAGE ANNOTATOR (VIA) Transferability.

### VirtualEnv Instructions
```
virtualenv GeneralEnv
GeneralEnv\Scripts\activate
conda.bat deactivate
```

### Install Requirememnts
```
pip install -r requirements.txt
```
### Generates JSONs
```
python generate_JSON.py
```
##### Note : You might want to read and change components from lines 62-92 of file generate_JSON.py first!

### Run LabelMe and Open DATA_PATH (Output JSONs) Directory
```
labelme
```