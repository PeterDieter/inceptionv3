import torch
import torch.optim as optim
from model import inception_v3, Inception3
from preProcess import Preprocess
from PIL import Image
import torch.nn as nn


cat_image = Image.open("images/cat_example2.jpg")
cat_image = Preprocess().preprocess_image_data([cat_image])

# Load Model
model =  Inception3()
model.load_state_dict(torch.load('bestmodel.pth'))
model.eval()

output = model(cat_image)
predicted_class = torch.argmax(torch.nn.functional.softmax(output, dim=1),1)

if predicted_class == 1:
    print("This is a cat")
else:
    print("This is a dog")