import torch
import torch.optim as optim
from model import inception_v3
from loadData import Data
from preProcess import Preprocess
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)
total_image_no = 1000
epochs = 25
batch_size = 28

# Load Model
model = inception_v3(pretrained=True)

# Load Data
cat_images = []
for i in range(1,total_image_no):
    cat_image = Image.open("training_set/training_set/cats/cat." + str(i) + ".jpg")
    cat_images.append(cat_image)

dog_images = []
for i in range(1,total_image_no):
    dog_image = Image.open("training_set/training_set/dogs/dog." + str(i) + ".jpg")
    dog_images.append(dog_image)


label_cats = torch.cat((torch.ones(total_image_no-1),torch.zeros(total_image_no-1)),0).type(torch.LongTensor)
label_dogs = torch.cat((torch.zeros(total_image_no-1),torch.ones(total_image_no-1)),0)
all_labels = torch.stack((label_cats, label_dogs)).transpose(0,1)



# PrecProcess data
cat_images = Preprocess().preprocess_image_data(cat_images)
dog_images = Preprocess().preprocess_image_data(dog_images)
all_images = torch.cat((cat_images, dog_images),0)

# create your optimizer and define loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_function = torch.nn.CrossEntropyLoss()


losses = []
accuracies = []
for i in range(epochs):
    rnd_indexes = torch.randint(len(all_images),[batch_size])
    image_batch = all_images[rnd_indexes]
    label_batch = label_cats[rnd_indexes]

    # Train the model
    optimizer.zero_grad()   # zero the gradient buffers
    output = model(image_batch)
    loss1 = loss_function(output[0], label_batch)
    loss2 = loss_function(output[1], label_batch)
    loss = loss1 + 0.4*loss2
    predicted_class = torch.argmax(torch.nn.functional.softmax(output[0], dim=1),1)
    accuracy = (predicted_class == label_batch).sum()/len(predicted_class)
    accuracies.append(accuracy)
    losses.append(loss)
    
    loss.backward()
    optimizer.step()    # Does the update
    print(accuracy)

torch.save(model.state_dict(), 'bestmodel.pth')

# Plot Accuracies
plt.plot(accuracies)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.show()

# Plot Losses
plt.plot(losses)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Cross Entropy Loss', fontsize=16)
plt.show()