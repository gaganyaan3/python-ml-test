import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import json

model = ResNet50(weights='imagenet')

img_path = '51663859411_8ce6e3b1e0_o.jpg' # The image to classify

img = cv2.imread(img_path)

img = cv2.resize(img, (224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

# Decode and display predictions
print(json.dumps(str(decode_predictions(preds, top=10)[0])))