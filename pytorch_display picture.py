


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import random
import numpy as np
from IPython.display import Image


'''
from subprocess import check_output
print("Here are the input datasets: ")
print(check_output(["ls", "../input"]).decode("utf8"))

print("Python Version: ")
'''
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};fig = plt.figure(figsize=(8,8));
'''
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(training_data));
    img = training_data[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[training_data[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()
'''
image,label = training_data[1]
plt.title(labels_map[label])
plt.imshow(image.squeeze(), cmap="gray")
plt.show()


'''
DATA_DIR = '../input/FashionMINST/test.out.npy'
X_train = np.load(DATA_DIR)
print(f"Shape of training data: {X_train.shape}")
print(f"Data type: {type(X_train)}")

data = training_data.astype(np.float64)
data = 255 * data
X_train = data.astype(np.uint8)

random_image = random.randint(0, len(X_train))
plt.imshow(X_train[random_image])
plt.title(f"Training example #{random_image}")
plt.axis('off')
plt.show()

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

dataiter=iter(train_dataloader)
print(dataiter)
images,labels=dataiter.next()

fig=plt.figure(figsize=(15,5))
for idx in np.arange(20):
    ax=fig.add_subplot(4,20/4,idx+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmap='gray')
    ax.set_title(labels[idx].item())
    fig.tight_layout()
    '''
