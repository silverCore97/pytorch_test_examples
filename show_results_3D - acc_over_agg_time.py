import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import Timer
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os, sys, sqlite3


#Check existence of DB and clear data
if not os.path.exists("ML_data.db"):
    print("Database does not exist. Please create the database file first.")
    sys.exit(0)

# Create connection to DB
connection = sqlite3.connect("ML_data.db")

# Create DB cursor
cursor = connection.cursor()

#Aquire number of n, m and epochs via aggregated data
nNodes=[]
mNodes=[]
epochs=0
accuracy=[]
time=[]

sql="SELECT DISTINCT n FROM Data"
cursor.execute(sql)
for dsatz in cursor:
    nNodes.append(dsatz[0])

sql="SELECT DISTINCT m FROM Data"
cursor.execute(sql)
for dsatz in cursor:
    mNodes.append(dsatz[0])

sql="SELECT MAX(epoch) FROM Data"
cursor.execute(sql)
for dsatz in cursor:
    epochs=dsatz[0]

print(nNodes,mNodes,epochs)

#Accuracy list
sql="SELECT accuracy FROM Data WHERE epoch = 10"
cursor.execute(sql)
for dsatz in cursor:
    accuracy.append(dsatz[0])

#Time list
sql="SELECT SUM(time) FROM Data GROUP BY n,m "
cursor.execute(sql)
for dsatz in cursor:
    time.append(dsatz[0])

#Reshape lists to 2D arrays
accuracy=np.array(accuracy)
time=np.array(time)
accuracy=accuracy.reshape(len(nNodes),len(mNodes))
time=time.reshape(len(nNodes),len(mNodes))

#New metric acc/time
metric=np.divide(accuracy,time)


#build input for wireframe
x=np.array(nNodes)
y=np.array(mNodes)
Y,X = np.meshgrid(x,y)
Z=np.array(metric)
print(Y)
print(Y)
print(Z)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.title("accuracy/time")
plt.xlabel('n')
plt.ylabel('m')
plt.show()
