#main.py

from config import apply_configurations
from feature_extraction import extract_features
from scene_description import generate_scene_description
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main(image_path):
    apply_configurations()
    features = extract_features(image_path)
    print("Extracted Features Shape:", features.shape)

    description = generate_scene_description(image_path)

    combined_output = {
        'features': features.numpy().flatten(),  # Flatten the features for easier handling
        'description': description  # Single description
    }

    return combined_output

if __name__ == "__main__":
    imgfile = "test2.jpg" 
    result = main(imgfile)

    img = Image.open(imgfile)
    img = np.array(img)

    plt.imshow(img)
    plt.axis('off') 
    plt.show()

    print("Description:")
    print(result['description'])  # Print the description directly
