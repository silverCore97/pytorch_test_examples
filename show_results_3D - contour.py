import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
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
sql="SELECT time FROM Data WHERE epoch = 10"
cursor.execute(sql)
for dsatz in cursor:
    time.append(dsatz[0])

#Reshape lists to 2D arrays
accuracy=np.array(accuracy)
time=np.array(time)
accuracy=accuracy.reshape(len(nNodes),len(mNodes))
time=time.reshape(len(nNodes),len(mNodes))


#build input for contour plot
x=np.array(nNodes)
y=np.array(mNodes)
Y,X = np.meshgrid(x,y)
X=X.flatten()
Y=Y.flatten()
Z=np.array(accuracy).flatten()
Z2=np.array(time).flatten()
print(X)
print(Y)
print(Z)

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
c=ax[0].tricontourf(X,Y,Z,20)
c2=ax[1].tricontourf(X,Y,Z2, 20) # choose 20 contour levels, just to show how good its interpolation is
fig.colorbar(c, ax=ax[0])
fig.colorbar(c2, ax=ax[1])
ax[0].plot(x,y, 'ko ')
ax[1].plot(x,y, 'ko ')
ax[0].set_title('Accuracy')
ax[1].set_title('Time')
plt.show()
'''
#build input for wireframe
x=np.array(nNodes)
y=np.array(mNodes)
Y,X = np.meshgrid(x,y)
Z=np.array(time)
print(Y)
print(Y)
print(Z)

fig = plt.figure()
#ax.contour(X, Y, Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
cntr2 = ax.tricontourf(X, Y, Z, levels=14, cmap="RdBu_r")
plt.xlabel('n')
plt.ylabel('m')
plt.show()'''
