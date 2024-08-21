import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).resize(target_size)
    image = np.array(image) / 255.0  
    return np.expand_dims(image, axis=0)  

def enhance_image(model, image_path, output_path):
    image = preprocess_image(image_path)
    enhanced_image = model.predict(image)[0]
    enhanced_image = (enhanced_image * 255).astype(np.uint8) 
    Image.fromarray(enhanced_image).save(output_path)
