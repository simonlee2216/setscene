# feature_extraction.py

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

def extract_features(image_path):
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img_tensor)

    return features

if __name__ == "__main__":
    # Test the feature extraction
    imgfile = "test1.jpg"  # Replace with your image file
    features = extract_features(imgfile)
    print("Extracted Features Shape:", features.shape)
