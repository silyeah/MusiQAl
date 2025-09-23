import os
import urllib.request

# Where to save the pretrained model
save_dir = os.path.join(os.path.dirname(__file__), "pretrained")
os.makedirs(save_dir, exist_ok=True)

url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
save_path = os.path.join(save_dir, "resnet18-5c106cde.pth")

if not os.path.exists(save_path):
    print(f"Downloading ResNet-18 weights from {url}...")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to {save_path}")
else:
    print(f"ResNet-18 weights already exist at {save_path}")


