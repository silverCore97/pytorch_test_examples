#This program saves the number of nodes in each hidden layer, current epoch, accuracy of
#that epoch and the time required for each epoch in a DB? file
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import Timer
import numpy as np
import os, sys, sqlite3

# Create connection to DB
connection = sqlite3.connect("ML_data.db")

# Create DB cursor
cursor = connection.cursor()

#Check existence of DB and clear data
if os.path.exists("ML_data.db"):
    sql = "DROP TABLE Data"
    cursor.execute(sql)
    connection.commit()

# Create table
sql = "CREATE TABLE Data(" \
      "n INTEGER, " \
      "m INTEGER, " \
      "epoch INTEGER, " \
      "accuracy REAL, " \
      "time REAL)"
cursor.execute(sql)



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
    def __init__(self,n=512,m=512):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( #Sequential network
            nn.Linear(28*28, n),#First hidden layer with n nodes
            nn.ReLU(),
            nn.Linear(n, m), #Second hidden layer with m nodes
            nn.ReLU(),
            nn.Linear(m, 10)
            
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
    # Save accuracy and average loss after an epoch in the variables.txt file
    return 100*correct


'''
#Create the file if non-existing and/ or clear the file
file=open("parameters.txt","w")
file.write("")
file.close()
file=open("parameters.txt","r+")
file.truncate(0)
file.close()
'''
epochs = 3   #Number of epochs = Nr. of iterations over the whole training set
results = []
it = 0    # it as iterator
n = 8    # We change the size of the first hidden layer [8, 16, 32, 64, 128, 256, 512]
m = 8
nRange=2
mRange=2
t=Timer.Timer()
for i in range(nRange):
    m=8
    for j in range(mRange):
        print(str(m))
        print(str(n))
        model = NeuralNetwork(n,m).to(device)
        print(model)
    
        loss_fn = nn.CrossEntropyLoss() #We choose cross-entropy loss as loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #Stochastic Gradient Descent with learning rate of 0.001
        
        for r in range(epochs):#For each epoch we require the data
            print(str(i))
            results.append([])
            #print(results)
            results[it].append(n)
            results[it].append(m)
            results[it].append(r+1)
            t.start() #We measure only training time
            
            print(f"Epoch {r+1}\n-------------------------------")      
            train(train_dataloader, model, loss_fn, optimizer)
            results[it].append(test(test_dataloader, model, loss_fn))
            t.stop()    #As the training is finished, we stop time measurement.
            diff=t.duration()
            results[it].append(diff)    #We save the results in a list
            it+=1
            
        print("Done!")        
        m*=2
        
    n*=2

#Results in DB
for l in results:
    print(l)
    if l==[]:
        print("extra bracket")
        break
    sql="INSERT INTO Data VALUES("+str(l[0])+","+str(l[1])+","+str(l[2])+","+str(l[3])+"," +str(l[4]) +")"
    cursor.execute(sql)
    connection.commit()






    '''
nNodes=[8]
for i in range(nRange-1):
    temp=nNodes[i]*2
    nNodes.append(temp)
    print(nNodes)
    
mNodes=[8]
for j in range(mRange-1):
    temp=mNodes[j]*2
    mNodes.append(temp)
    print(mNodes)

print(results)

#build input for wireframe
x=np.array(nNodes)
y=np.array(mNodes)
Y,X = np.meshgrid(y,x)
Z=np.array(results)
print(Y)
print(Y)
print(Z)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.xlabel('n')
plt.ylabel('m')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot a basic wireframe.
ax.plot_wireframe(nNodes, mNodes, results, rstride=4, cstride=4)

plt.show()


#Transform the string pair list into 2 float list
nodes=[]
t2=[]
for l in res:
    nodes.append(float(l[1]))
    t2.append(float(l[0]))


# Open accuracy text file and transform it into float list acc
file=open(r"acc.txt","r")
list=file.read()
list2 = list.split(" ")
for l in list2:
    if l!="":
        acc.append(float(l))
#Only the accuracies of the last epoch are relevant
acc2=[]
for a in range(epochs-1,len(acc),epochs):
    print(acc[a])
    acc2.append(acc[a])

#Construct graph
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Time and accuracy')
ax1.plot(nodes,t2)
print(acc2)
ax2.plot(nodes,acc2)
plt.show()

plt.figure(1)
plt.subplot(211)  
plt.plot(nodes,t2)
plt.subplot(221)
plt.plot(nodes,acc)
plt.show()
'''













