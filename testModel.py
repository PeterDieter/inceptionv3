import torch
import torch.optim as optim
from model import inception_v3, Inception3
from preProcess import Preprocess
from PIL import Image
import torch.nn as nn

torch.manual_seed(1)
total_image_no = 60

# Load Model
model =  Inception3()
model.load_state_dict(torch.load('bestmodel.pth'))
model.eval()

# Load Data
cat_images = []
for i in range(4001, 4000 + total_image_no):
    cat_image = Image.open("test_set/test_set/cats/cat." + str(i) + ".jpg")
    cat_images.append(cat_image)

dog_images = []
for i in range(4001, 4000 + total_image_no):
    dog_image = Image.open("test_set/test_set/dogs/dog." + str(i) + ".jpg")
    dog_images.append(dog_image)


label_cats = torch.cat((torch.ones(total_image_no-1),torch.zeros(total_image_no-1)),0).type(torch.LongTensor)
label_dogs = torch.cat((torch.zeros(total_image_no-1),torch.ones(total_image_no-1)),0)
all_labels = torch.stack((label_cats, label_dogs)).transpose(0,1)

# PrecProcess data
cat_images = Preprocess().preprocess_image_data(cat_images)
dog_images = Preprocess().preprocess_image_data(dog_images)
all_images = torch.cat((cat_images, dog_images),0)



# Make Predictions
output = model(all_images)
predicted_class = torch.argmax(torch.nn.functional.softmax(output, dim=1),1)
accuracy = (predicted_class == label_cats).sum()/len(predicted_class)

print(accuracy)