import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import time

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Ones Tensor: \n {ones_tensor} \n")
#print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.ones(4, 4)
#print('First row: ', tensor[0])
#print('First column: ', tensor[:, 0])
#print('Last column:', tensor[..., -1])
tensor[:,1] = 0
#print(tensor)

print(tensor, "\n")
tensor.add(5)
print(tensor)
tensor.add_(5)
print(tensor)























