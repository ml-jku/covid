#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at
        
Dataset and architectures for Logistic Regression and Self-Normalizing Neural Network
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
   
class myDataset(Dataset):
    
    '''
    Dataset for Logistic Regression and Self-Normalizing Neural Network
    '''  
    
    def __init__(self, data, start_feature, label_column = 'corona'):
        
        """
        Parameters
        ----------
        data: Pandas Dataframe containing the data.
        start_feature: the keys of the dataframe are used for the model, 
            starting from the 'start_feature' index.
        label_column: Indicating what is the task. COVID_19 diagnosis prediction 
            with 'corona' or mortality prediciton with 'death_in_hospital'. 
            Indicates the column of the dataframe containing the label. 
        """
        
        self.datacolumns = data.keys()[start_feature:]
        
        self.data = torch.FloatTensor(data[self.datacolumns].values)
        self.labels = torch.FloatTensor(data[label_column].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        """
        Parameters
        ----------
        idx: Index of sample. 
            
        Returns
        ----------
        self.data[idx]: Return the sample at index 'idx'.
        self.labels[idx]: Return the label at index 'idx'.
        """
        
        return self.data[idx], self.labels[idx]
    

class logisticRegression(torch.nn.Module):
    
    '''
    Logistic Regression Model
    '''
    
    def __init__(self, inputSize, outputSize):
        
        """
        Parameters
        ----------
        inputSize: Number of features of input.
        outputSize: Dimension of output.
        """
        
        super(logisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        
        """
        Parameters
        ----------
        x: Input. Batch of samples with blood test features.
        
        Returns
        ----------
        self.linear1(x): Predictions of the Logistic Regression model for input x.
        """
        
        return self.linear1(x)


class SNN(torch.nn.Module):
    
    '''
    Self-Normalizing Neural Network Model
    '''
    
    def __init__(self, inputSize, outputSize, intermediate_size, n_layers, dropout):
        super(SNN, self).__init__()
        
        """
        Parameters
        ----------
        inputSize: Number of features of input.
        outputSize: Dimension of output.
        intermediate_size: Number of neurons in intermediate layers.
        n_layers: Number of layers in the SNN.
        dropout: Probability of alpha-dropout.
        """
        
        self.intermediate_size = intermediate_size
        
        self.linear1 = torch.nn.Linear(inputSize, intermediate_size)
        self.linear1.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear1.weight.shape)))
        
        self.linear3 = torch.nn.Linear(intermediate_size, outputSize)
        self.linear3.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear3.weight.shape)))
        
        self.alphadropout = torch.nn.AlphaDropout(p = dropout)
        self.relu = torch.nn.ReLU()
        
        if n_layers >= 1:
            self.linear2a = torch.nn.Linear(intermediate_size, intermediate_size)
            self.linear2a.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear2a.weight.shape)))
            
            self.net1 = torch.nn.Sequential(self.linear1,
                                            torch.nn.SELU(),
                                            self.linear2a, 
                                            torch.nn.SELU(), 
                                            self.linear3)
            
        if n_layers > 2:
            self.linear2b = torch.nn.Linear(intermediate_size, intermediate_size)
            self.linear2b.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear2b.weight.shape)))
            self.linear2c = torch.nn.Linear(intermediate_size, intermediate_size)
            self.linear2c.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear2c.weight.shape)))
            
            self.net3 = torch.nn.Sequential(self.linear1,
                                            torch.nn.SELU(),
                                            self.linear2a,
                                            torch.nn.SELU(),
                                            self.linear2b,
                                            self.alphadropout,
                                            torch.nn.SELU(),
                                            self.linear2c,
                                            torch.nn.SELU(),
                                            self.linear3)
            
        if n_layers > 3:
            self.linear2d = torch.nn.Linear(intermediate_size, intermediate_size)
            self.linear2d.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear2d.weight.shape)))
            self.linear2e = torch.nn.Linear(intermediate_size, intermediate_size)
            self.linear2e.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear2e.weight.shape)))
            self.linear2f = torch.nn.Linear(intermediate_size, intermediate_size)
            self.linear2f.weight.data.normal_(0.0, 
                            np.sqrt(1 / np.prod(self.linear2f.weight.shape)))
            
            self.net6 = torch.nn.Sequential(self.linear1,
                                            torch.nn.SELU(),
                                            self.linear2a,
                                            torch.nn.SELU(),
                                            self.linear2b,
                                            self.alphadropout,
                                            torch.nn.SELU(),
                                            self.linear2c,
                                            torch.nn.SELU(),
                                            self.linear2d,
                                            torch.nn.SELU(),
                                            self.alphadropout,
                                            self.linear2e,
                                            torch.nn.SELU(),
                                            self.linear2f,
                                            torch.nn.SELU(),
                                            self.linear3)

        if (n_layers == 1):
            self.net = self.net1
        elif (n_layers == 3):
            self.net = self.net3
        else:
            self.net = self.net6

    def forward(self, x):
        
        """
        Parameters
        ----------
        x: Input. Batch of samples with blood test features.
        
        Returns
        ----------
        self.net(x): Predictions of the Self-Normalizing Neural Network for input x.
        """
        
        return self.net(x)
    


