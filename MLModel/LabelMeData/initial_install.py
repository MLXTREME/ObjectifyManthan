import wget, os

file_url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl"

if not os.path.exists("pointrend_resnet50.pkl"):
    wget.download(file_url)
    print("Created Model!")
else:
    print("Model Already Installed!")
