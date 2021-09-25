import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
import matplotlib.mlab as mlab
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


#Interpolation of new values
xi=[]
yi=[]
for i in range(128):
    xi.append(8+8*i)
    yi.append(8+8*i)
triang = tri.Triangulation(X, Y)
interpolator = tri.LinearTriInterpolator(triang, Z)
Yi, Xi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)
interpolator2 = tri.LinearTriInterpolator(triang, Z2)
z2i = interpolator2(Xi, Yi)



fig = plt.figure()

# First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(Xi, Yi, zi, cmap=cm.coolwarm)
fig.colorbar(surf, ax=ax)
ax.set_title('Accuracy')

# Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax.plot_surface(Xi, Yi, z2i, cmap=cm.coolwarm)
fig.colorbar(surf2, ax=ax)
ax.set_title('Time')
plt.show()
'''
# As seperate plots
# Accuracy
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(Xi, Yi, zi, cmap=cm.coolwarm)
fig.colorbar(surf, ax=ax)
ax.set_title('Accuracy')
plt.show()

# Time
fig = plt.figure()
ax = Axes3D(fig)
surf2 = ax.plot_surface(Xi, Yi, z2i, cmap=cm.coolwarm)
fig.colorbar(surf2, ax=ax)
ax.set_title('Time')
plt.show()
'''


