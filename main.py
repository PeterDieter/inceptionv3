import torch
import urllib
from PIL import Image
from torchvision import transforms
from model import inception_v3
from loadData import Data
from preProcess import Preprocess

# Load Model
model = inception_v3(pretrained=True)
# Set model to evaluation mode
model.eval()

# Download an example image from the pytorch website
filename = ("dog.1.jpg")
input_image = Data().load_single_image(filename)

input_batch = Preprocess().preprocess_image_data(input_image)
print(input_batch.size())
# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
  output = model(input_batch)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))
print(torch.argmax(torch.nn.functional.softmax(output[0], dim=0)))
print(torch.max(torch.nn.functional.softmax(output[0], dim=0)))