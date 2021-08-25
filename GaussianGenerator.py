#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:16:35 2021

@author: emanuele
"""

"""
This code generate sequences of indipendent gaussian numbers, group them into arrey and store them into a file; each sequence should be stored in a different file
nota in questo modo le estrazioni lungo le componenti sono indipendenti (corrisponde al caso di matrice di covarianza diagonale); una valida alternativa potrebbe essere usare
torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None)
vedi come funziona
"""

"""
import random
import os
import numpy as np

mu = 100
sigma = 50

#FolderPath = './data/'+ 'mu_'+ str(mu) + '_sigma_' + str(sigma)
FolderPath = './data/'+ str(0)

if not os.path.exists(FolderPath):
    os.mkdir(FolderPath) 
    
    
DatasetSize = 10000
DataSize = 2

#SEED INITIALIZATION MUST BE ADDED
random.seed(0)

for i in range(0, DatasetSize):
    arr = np.zeros(DataSize)
    for j in range(0, DataSize):
        arr[j] = random.gauss(mu, sigma)    
    with open(FolderPath + "/"+ str(i) + '.txt', "a") as f:
        np.savetxt(f, arr, delimiter = ',')
        
        
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import os
import numpy as np

DatasetSize = 100000


"""
#pytorch implementation (sampling with MultivariateNormal)
#mean = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
mean = torch.Tensor([1., 1., 1., 1.,1., 1., 1., 1.,1., 1.])
DataSize = len(mean)
#be careful on the choose of the covariance matrix, make sure that it is  positive definite (By using symmetric matrices)
cov = 2.5*torch.eye(len(mean))
"""

#numpy implementation (sampling from random module)

#mean = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
#mean = np.array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])
#mean = np.array([16., 16., 16., 16., 16., 16., 16., 16., 16., 16.])
mean = np.array([1.7, 1.7])
#mean = -mean
print(mean)
cov = np.zeros((len(mean), len(mean)))
print(cov)
np.fill_diagonal(cov, 4.)

print(cov)

FolderPath = './data'

if not os.path.exists(FolderPath):
    os.mkdir(FolderPath) 

FolderPath = './data/'+ str(0)

if not os.path.exists(FolderPath):
    os.mkdir(FolderPath) 
    
"""    
#pytorch implementation (sampling with MultivariateNormal)
distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
for i in range(0, DatasetSize):
    torch.save(distrib.sample(), FolderPath +'/' + str(i) +'.pt')
    print(distrib.sample())
"""

x = np.random.multivariate_normal(mean, cov, DatasetSize)
"""
#checks
meancheck = np.zeros(len(mean))
covcheck = np.zeros((len(mean), len(mean)))
for j in range (0,DatasetSize):
    meancheck += x[j]
    covcheck += np.dot(   x[j][..., None],  np.transpose( x[j][..., None]) ) 
a = covcheck/DatasetSize
meancheck = meancheck/DatasetSize
b = np.dot(meancheck[..., None], np.transpose(meancheck[..., None]))


print("pre", covcheck/DatasetSize)
print("dur", - np.dot(meancheck[..., None], np.transpose(meancheck[..., None])))
c = a-b
print("post", c )
#print(torch.from_numpy(x[0].astype(float)))
"""
for i in range(0, DatasetSize):
    torch.save(torch.from_numpy(x[i].astype(float)), FolderPath +'/' + str(i) +'.pt')