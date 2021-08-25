#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:48:18 2021

@author: emanuele
"""

"""
Here a simple linea network (in which input and output are directly connected without intermediate hidden layers) is implemented; input are given by 
random strings generated following a gaussian distribution in each component
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F


import os
#manage the input of values from outside the script 
import argparse

import numpy as np
import math

p = argparse.ArgumentParser(description = 'Sample index')
p.add_argument('SampleIndex', help = 'Sample index')
p.add_argument('FolderName', type = str, help = 'Name of the main folder where to put the samples folders')

args = p.parse_args()

print('il primo parametro passato (indice del campione) è: ', args.SampleIndex)
print('il secondo parametro passato (cartella per output storage) è: ', args.FolderName)

#we first create the folder associated to the sample and then save inside the folder all the files
#iniziamo col creare il path per la cartella da creare
FolderPath = './'+ args.FolderName + '/Sample' + str(args.SampleIndex)
print('La cartella creata per il sample ha come path: ', FolderPath)
if not os.path.exists(FolderPath):
    os.mkdir(FolderPath) 



#we start creating our own dataset from the files created by "GaussianGenerator.py" each file contain a single tensor data; files are divided in folder according to their belonging class


class GaussBlobsDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        self.class_map = {}

        for family in os.listdir(data_root):
            family_folder = os.path.join(data_root, family)

            for sample in os.listdir(family_folder):
                sample_filepath = os.path.join(family_folder, sample)
                self.samples.append((family, torch.load(sample_filepath)))
            #create a mapping for the classes
            self.class_map[family] = int(family)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_name, data = self.samples[idx]
        class_id = self.class_map[class_name]
        
        #we don't need to convert label in tensor here; dataloader will do it for us (creating a batch tensor label)
        #class_id = torch.tensor([class_id])
        
        #return self.samples[idx]
        return data, class_id


"""
if __name__ == '__main__':
    dataset = TESNamesDataset('./data')
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, batch in enumerate(dataloader):
        print(i, batch)
        
"""

dataset = GaussBlobsDataset('./data')
num_data = len(dataset)
print("total number of samples ", num_data)
# percentage of training set to use as validation
valid_size = 0.4
batch_size = 20
num_workers = 0
#setting Learning rate
learning_rate = 0.01



num_classes=0
for family in os.listdir('./data'):
    num_classes+=1
print("number of classes: ", num_classes)
#print(dataset[0])

input_len = len(dataset[0][0])
print("input size: ", input_len)




DebugFile = open("./DebugChecks.txt", "a")
#creo i file dove storare gli output dei vari print
info = open("./infogenerali.txt", "a") 


#FLAG VARIABLES
#more possibilities for the dynamic (for now Classical minibatch SGD AND PCNSGD have been implemented)
Dynamic = 'SGD'
if (Dynamic=='SGD'):
    print("the simulation use minibatch SGD as dynamic", file = info)
elif (Dynamic=='GD'):
    print("the simulation use a dynamic similar to minibatch SGD but with classes terms decoupled and normalized with gradient norm (calculated over the class element of the batch)", file = info)
else:
    print('Improper value for the Dynamic flag variable', file = info)


# Get cpu or gpu device for training #MODIFIED BLOCK
#for now we use only cpu but we can change uncommenting the following lines and comment the one following

#device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using {} device".format(device))

#comment the following line when uncomment the 2 above
device = "cpu"




indices = list(range(num_data))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_data))
train_index, valid_index = indices[split:], indices[:split]
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)
# prepare data loaders
train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)





n_epochs = 100

NSteps = 40

MaxStep = n_epochs*len(train_loader.sampler)/batch_size
#creiamo una scala equispaziata logaritmicamente (in base 2 ad esempio)
Times = np.logspace(3, np.log2(MaxStep),num=NSteps, base=2.) 
Times = np.rint(Times).astype(int)
print('I tempi usati per la simulazione sono', Times, flush=True, file = info)

IterationCounter = 0
TimesComponentCounter = 0
#storage variables

TrainLoss = []
TrainAcc = []
PCGAngles = np.full((int(num_classes*(num_classes-1)/2), (n_epochs + 1)), 10.)

TP = torch.zeros((num_classes , NSteps))
TN = torch.zeros((num_classes, NSteps))
FP = torch.zeros((num_classes, NSteps))
FN = torch.zeros((num_classes, NSteps))
TotP = torch.zeros((num_classes, NSteps))

SteepnessVectorNorm = np.zeros(NSteps)
DiffVectorNorm = np.zeros(NSteps)





#define the network architecture; a linear connection between input and output (without presence of intermediate hidden layers)


class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super(Net,self).__init__()
        # number of hidden nodes in each layer (512)
        self.num_input = n_in
        self.num_classes = n_out

        self.fc1 = nn.Linear(self.num_input, self.num_classes)
        
        #weights initialization (this step can also be put below in a separate def)
        nn.init.xavier_normal_(self.fc1.weight)

    #I return from the forward a dictionary to get the output after each layer not only the last one
    #this is useful for example in the inter-classes angles (calculated throught the scalar product between inner layers representation)
    def forward(self,x):
        x = x.view(-1,self.num_input)
        Out = self.fc1(x)       
        return Out
# initialize the NN
model = Net(input_len, num_classes).to(device) #MODIFIED LINE
#print(model) 


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)




TrainClassesLoss = np.zeros((num_classes, (n_epochs + 1)))

#mean and covariance variables
mean1 = np.zeros(input_len)
mean2 = np.zeros(input_len)
cov1 = np.zeros((input_len, input_len))
cov2 = np.zeros((input_len, input_len))
cov = np.zeros((input_len, input_len))
covInv = np.zeros((input_len, input_len))
n0 = 0
n1 = 0

for data,label in train_loader:
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    
    output = model(data.float())

    #storing mean representation for the classes to use for the definition of the angles between classes

    for im in range(0, len(label)):
        TrainClassesLoss[label[im]][0] += criterion(output[im].expand(1, num_classes),(label[im]).expand(1)).item()

        
        #compute mean and covariance matrix
        if(label[im]==0):
            n0 +=1
            #print(data[im].clone().detach().numpy())
            mean1 += data[im].clone().detach().numpy()
            cov1 += np.dot(  data[im].clone().detach().numpy()[..., None],  np.transpose(data[im].clone().detach().numpy()[..., None]) )    
        elif(label[im]==1):
            n1 +=1
            mean2 += data[im].clone().detach().numpy()
            cov2 += np.dot(  data[im].clone().detach().numpy()[..., None],  np.transpose(data[im].clone().detach().numpy()[..., None]) ) 

mean1 = mean1/n0
mean2 = mean2/n1
print(mean1)
print(mean2)

cov1 = cov1- np.dot( mean1[..., None], np.transpose(mean1[..., None]))*n0      
#print(cov1)
cov2 = cov2 - np.dot( mean2[..., None], np.transpose(mean2[..., None]))*n1  
#print(cov1)

cov = (cov1 + cov2)/(n0+n1)
print(cov)
#inverse of covariance matrix
covInv = np.linalg.inv(cov)

LDASteepness = np.dot(covInv, (mean2-mean1))

print("steepest modulus parameter: ", np.linalg.norm(LDASteepness))
print(np.dot(covInv, (mean1-mean2)[..., None]))
print((mean1-mean2))
DeltaSquare = np.dot(np.transpose((mean1-mean2)[..., None]),  np.dot(covInv, (mean1-mean2)[..., None]) )
print(DeltaSquare)
inaccuracy = math.exp(-DeltaSquare/4)*2/(math.sqrt(math.pi*DeltaSquare))
print("predicted error: ", inaccuracy)
uncertainty = math.exp(-DeltaSquare/4)*4/(math.sqrt(math.pi)*(DeltaSquare**(3/2)))
print("boundary ", inaccuracy + uncertainty, inaccuracy - uncertainty)


#store the relevant parameter in one array (because we want to save them in a file and np.savetxt accept 1-D or 2-D array)
RelevantParameters = np.zeros(4)
RelevantParameters[0] = DeltaSquare
RelevantParameters[1] = inaccuracy
RelevantParameters[2] = uncertainty
RelevantParameters[3] = np.linalg.norm(LDASteepness)

#save more important parametric measures in files
with open(FolderPath + "/InvCov.txt", "w") as f:
    np.savetxt(f, covInv, delimiter = ',')

with open(FolderPath + "/DeltaSq_Inaccuracy_UncertUpperBond.txt", "a") as f:
    np.savetxt(f, RelevantParameters, delimiter = ',')



#training

for epoch in range(n_epochs):
    
    #check variables to see the consistence between theoretical prediction of lda and actual learning parameters
    par0 = np.zeros(input_len)
    par1 = np.zeros(input_len)
    nstep = 0
    
    
    model.train() # prep model for training
    
    #reset Gradient Norm (calculated for each epoch)
    total_norm = 0
    
    
    
    #if we perform SGD we just collect data in a log-equispaced time scale
    if (Dynamic=='SGD'):
        for data,label in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
    
            
            output = model(data.float())
            
            #print(output)
            #print(label)
            
            
            #storing mean representation for the classes to use for the definition of the angles between classes
        
            for im in range(0, len(label)):
                TrainClassesLoss[label[im]][epoch+1] += criterion(output[im].expand(1, num_classes),(label[im]).expand(1)).item()
       
            
            # calculate the loss
            loss = criterion(output,label)
            print(loss, flush=True, file = DebugFile)
            
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()








            IterationCounter +=1
            if (IterationCounter == Times[TimesComponentCounter]):
                
 
            
 
                par0 = np.zeros(input_len)
                par1 = np.zeros(input_len)
            
    
                filterVar = 0
                for p in model.parameters():
                    #print("la forma", p.shape)
                    if filterVar==0:
                        par0 += p[0].detach().numpy()
                        par1 += p[1].detach().numpy()
                    filterVar = 1
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2   
                WeightSteepness = par1-par0      
                #print( par0, par1)  
                DiffVector = WeightSteepness - LDASteepness
                
                SteepnessVectorNorm[TimesComponentCounter] = np.linalg.norm(WeightSteepness)
                DiffVectorNorm[TimesComponentCounter] = np.linalg.norm(DiffVector)
                
                print("Steepest weight estimation ", np.linalg.norm(WeightSteepness))
                print("norm of the difference vector ", np.linalg.norm(DiffVector))
 
                
                #print(IterationCounter)
                # monitor losses
                train_loss = 0
                valid_loss = 0
                
                # variables for training accuracy
                TrainCorrect = 0
                TrainTotal = len(train_loader.sampler)    
                ValCorrect = 0
                ValTotal = len(valid_loader.sampler) 
                
                model.eval()  # prep model for evaluation
                #we evaluate the training and test set at the times corresponding to the logarithmic equispaced steps
                for dataval,labelval in train_loader:
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(dataval.float())
    
                    
                    # calculate the loss
                    loss = criterion(output,labelval)
    
                    # update running validation loss (loss.item() ü l media della loss sul batch, lo rimoltiplico per il numero di batch per ottenere un oggetto estensivo (in modo che sia correttamente normalizzato sotto, con il numero di campioni))
                    train_loss += loss.item() * dataval.size(0)



                    _, TrainPred = torch.max(output, 1)
                    TrainCorrect += (TrainPred == (labelval)).sum().item()
    
                    #calculating TotP = TP + FP, TP, FP, FN
                    for i in range(0, len(labelval)):
                        #TotP[TrainPred[i].item()][TimesComponentCounter] +=1
                        if (TrainPred[i].item() == (labelval[i].item())):
                            TP[TrainPred[i].item()][TimesComponentCounter] +=1 
                        elif(TrainPred[i].item() != (labelval[i].item())):
                            FP[TrainPred[i].item()][TimesComponentCounter] +=1 
                            FN[(labelval[i].item())][TimesComponentCounter] +=1 
                    
                    
                    
                    
                
                TrainLoss.append(train_loss / len(train_loader.sampler))
                print('.train loss', train_loss / len(train_loader.sampler) ,'train accuracy', 100*TrainCorrect / TrainTotal, flush=True, file = info)
                TrainAcc.append(100*TrainCorrect / TrainTotal)
                TimesComponentCounter+=1
                print("1-accuracy ", (1-TrainCorrect / TrainTotal) )







        
            #COMPUTE GRADIENT NORM AT EACH EPOCH: ()
            #Tensor.grad: This attribute is None by default and becomes a Tensor the first time a call to backward() computes gradients for self. The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it.
            #y.data.norm() == y.data.norm(2)  is equivalent to  torch.sqrt(torch.sum(torch.pow(y, 2)))
            
            #NOTA: ERRORE CONCETTUALE, non puoi continuare a sommare norme quadre su diversi batch (diversi vettori); puoi calcolare la media o effettuare il calcolo solo per alcuni batch (equispaziati in scala log)

            """
            filterVar = 0
            for p in model.parameters():
                #print("la forma", p.shape)
                if filterVar==0:
                    par0 += p[0].detach().numpy()
                    par1 += p[1].detach().numpy()
                filterVar = 1
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2   
                
            nstep +=1   
            """
            # perform a single optimization step (parameter update)
            optimizer.step() 
            
        #print("EPOCH ", epoch, par0/nstep, par1/nstep)
        #print("EPOCH ", epoch, par0, par1)
        
        





        
    elif(Dynamic == 'GD'):
        
        Norm = np.zeros(num_classes)
        
        GradCopy = [[] for i in range(num_classes)]

        
        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        
        #reset Gradient Norm (calculated for each epoch)
        total_norm = 0

        for data,label in train_loader:
            
    

    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            for i in range(0, len(label)):

                #CNN layers expect a batch of images as input, so I will give as input batches of single element using .unsqueeze(0) 
                output = model(data[i].float().unsqueeze(0))
                loss = criterion(output.expand(1, num_classes),(label[i]).expand(1))


                #Store the last layer mean representation and per classes loss function
                TrainClassesLoss[label[i]][epoch+1] += loss.item()
                    
                
                loss.backward()
                ParCount = 0
                if not GradCopy[label[i]]:
                    for p in model.parameters():                                                        
                        GradCopy[label[i]].append(p.grad.clone())                                
                elif GradCopy[label[i]]:
                    for p in model.parameters(): 
                        GradCopy[label[i]][ParCount] = GradCopy[label[i]][ParCount].clone() + p.grad.clone()
                        ParCount +=1
                optimizer.zero_grad()
            #putting gradient to 0 before filling it with the per class normalized sum
            optimizer.zero_grad()
            

            
            
            
    
            # update running training loss
            #train_loss += loss.item() * data.size(0) #NOTA:QUESTO È UTILE PER IL CALCOLO DELLA LOSS SU UN'EPOCA (METTI A COMMENTO SE CALCOLI SU SINGOLI STEP)

                        
            
            
            
            
            

            
                
        
        
        #we end GD dynamic calculating the per-class norm, normalizing PC gradient, sum them and update the weight following the direction of the obtained vector
        #also the norm gradient on the whole dataset is calculated (summing gradient' vectors of all classes and calculating the corresponding norm)
        TotGrad = []
     
        train_loss=0
        for index in range(0, num_classes):
            #compute the total loss function as the sum of all class losses
            train_loss += TrainClassesLoss[index][epoch+1]            
            TGComp=0
            
            """
            for alpha in range(0, num_classes):
                print("pri", GradCopy[alpha]) 
            """
            for obj in GradCopy[index]:
                Norm[index] += (torch.norm(obj.clone()).detach().numpy())**2
                """
                print(obj)
                print(torch.norm(obj.clone()))
                print("torch.norm for index", index, (torch.norm(obj.clone()).detach().numpy())**2 )
                """
                #filling the total gradient
                if(index==0):
                    TotGrad.append(obj.clone())
                else:
                    TotGrad[TGComp] += obj.clone()
                    TGComp+=1
                
            Norm[index] = Norm[index]**0.5
            """
            for alpha in range(0, num_classes):
                print("dur", GradCopy[alpha])            
            """
            ParCount = 0
            for p in model.parameters():
                p.grad += torch.div(GradCopy[index][ParCount].clone(), len(train_loader.sampler))
                ParCount +=1   
            """
            for alpha in range(0, num_classes):
                print("dopo", GradCopy[alpha])
            """
        print("Norm of class gradient: ", Norm)
        
        TrainLoss.append(train_loss / len(train_loader.sampler))        
        #calculating the total gradient norm
        for obj in TotGrad:
            total_norm += (torch.norm(obj.clone()).detach().numpy())**2
            
        AngleIndex=0
        for i,j in((i,j) for i in range(num_classes) for j in range(i)):
            #calculate the cos(ang) as the scalar product normalized with the l2 norm of the vectors
            ScalProd=0
            TGComp=0
            for obj in GradCopy[i]:
                """
                print("oggetti del prodotto scalare")
                print(obj.detach().numpy())
                print(GradCopy[j][TGComp].detach().numpy())
                """
                ScalProd += np.sum(np.multiply(GradCopy[j][TGComp].detach().numpy(), obj.detach().numpy()))
                #print("scal prod", np.sum(np.multiply(GradCopy[j][TGComp].detach().numpy(), obj.detach().numpy())))
                TGComp +=1
            #print("cos is ", ScalProd/(Norm[i]*Norm[j]))
            PCGAngles[AngleIndex][epoch+1] = math.acos(ScalProd/(Norm[i]*Norm[j]))
            AngleIndex+=1 
                
        
        # perform a single optimization step (parameter update)
        optimizer.step()    
        
        
        filterVar = 0
        for p in model.parameters():
            #print("la forma", p.shape)
            if filterVar==0:
                par0 += p[0].detach().numpy()
                par1 += p[1].detach().numpy()
            filterVar = 1
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2   
            
        nstep +=1    
        # perform a single optimization step (parameter update)
        optimizer.step() 
        
    #print("EPOCH ", epoch, par0/nstep, par1/nstep)        
        
        
    #CALCOLA LA HARDNESS (COME STEEPNESS DELLA LOGISTIC), MISURA ANCHE ANGOLI, NORME DELLE RAPPRESENTAZIONI E LOSS ()
    #CONSISTENCE CHECK: compare the stepness of the logistic (calculated from the weights of the network) with the one calculated from distribution parameters
    



with open(FolderPath + "/time.txt", "w") as f:
    np.savetxt(f, Times, delimiter = ',')


TempTrainingLoss=TrainLoss
with open(FolderPath + "/TrainingLoss.txt", "a") as f:
    np.savetxt(f, np.array(TempTrainingLoss), delimiter = ',')


with open(FolderPath + "/TrainAcc.txt", "a") as f:
    np.savetxt(f, TrainAcc, delimiter = ',')  

with open(FolderPath + "/TrainClassesLoss.txt", "a") as f:
    np.savetxt(f, TrainClassesLoss, delimiter = ',')    
    
with open(FolderPath + "/SteepnessNorm.txt", "a") as f:
    np.savetxt(f, SteepnessVectorNorm , delimiter = ',')    

with open(FolderPath + "/DiffSteepnessNorm.txt", "a") as f:
    np.savetxt(f, DiffVectorNorm , delimiter = ',')  
    
    











DebugFile.close()