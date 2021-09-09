import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import Timer

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module): #Input 28*28 picture in black and white
    def __init__(self,n=512):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( #Sequential network
            nn.Linear(28*28, n),#First hidden layer with n nodes
            nn.ReLU(),
            nn.Linear(n, 512), #Second hidden layer with 512 nodes
            nn.ReLU(),
            nn.Linear(512, 10)
            
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



#Training data using a model, a loss function and an optimizer.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:  #Every 100 batches = 6400 data items
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#Test data by comparing the values predicted by the model with the actual values
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")





epochs = 1   #Number of epochs = Nr. of iterations over the whole training set
results = []
n=32    # We change the size of the first hidden layer [32, 64, 128, 256, 512]
t=Timer.Timer()
for i in range(5):
    
    model = NeuralNetwork(n).to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss() #We choose cross-entropy loss as loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #Stochastic Gradient Descent with learning rate of 0.001
    
    t.start() #We measure only training time
    
    for r in range(epochs):
        print(f"Epoch {r+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    t.stop()    #As the training is finished, we stop time measurement.
    diff=t.duration()
    results.append([diff,n])    #We save the results in a list
    n*=2


#Write results in a file
file = open(r"results.txt","w")
text=""
for r in range(len(results)):
    text += str(results[r][0]) +" "+ str(results[r][1]) +"\n"

file.write(text)
file.close()

#Open file and retrieve results
res = []
file = file = open(r"results.txt","r")
list=file.readlines()
for l in list:
    res.append(l.split(" "))

#Transform the string pair list into 2 float list
nodes=[]
t2=[]
for l in res:
    nodes.append(float(l[1]))
    t2.append(float(l[0]))

#print(nodes)    For testing
#print (t2)
#print(res)

#Construct graph
plt.figure(1)
plt.subplot(211)  
plt.plot(nodes,t2)
plt.show()






'''
    
# Printing results as text
for i in range(len(res)):
    print(str(epochs)+" epochs took {0:.2f} seconds with {1:d} nodes for the first hidden layer".format(results[i][0],results[i][1]))

#Saving model for future use
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#Loading the model again
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

#Dictionary for the label values 0 to 9
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]   #First test item tested with model
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

'''

    









