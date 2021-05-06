#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at
        
Initialize, train and validate the models.
"""

import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
import multiprocessing as mp
import copy as copy
from architectures import logisticRegression, SNN
from evaluate_model import test_model, evaluate_test
from preprocessing import get_weights

def train_model(selected_model, start_feature, keys, hyperparamlist, data, 
                train_indices, fulldataset = None, trainloader = None, 
                valloader = None, k = 0, k_hyper = 0, k_val = 0, 
                val_indices = None, criterion = 'BCEWithLogitsLoss', 
                sampling_2019_weights  = 1, sampling_2020_weights = [1,1,1,1], 
                label_column = 'corona', test_months = [11,12], device = 'cpu'):
    
    '''
    Initialize, train and validate the model.
    Return the ROC AUC on the basis of the validation set and the trained model
    
    Parameters
    ----------
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    start_feature: the keys of the dataframe are used for the model, starting 
        from the 'start_feature' index.
    keys: The keys (names) of the hyperparameters stored in the dictionary.
    hyperparamlist: A list containing all possible hyperparameters. This list
        can be iterated in grid search.
    data: Pandas dataframe containing the data set.
    train_indices: The indices of the training set.
    fulldataset: Torch dataset containing all data (required for all models but 
        NOT for 'LOGREG' and 'SNN').
    trainloader: Torch dataloader for training (required for LOGREG and SNN).
    valloader: Torch dataloader for validation (required for LOGREG and SNN).
    k: Counter of the test loop.
    k_hyper: Counter of the loop for the hyperparametersearch.
    k_val: Counter for the validation loop.
    val_indices: The indices of the validation set (optional).
    criterion: Criterion to calculate the loss between prediction and target.
    sampling_2019_weights: The weight of the samples of the 2019 cohort is 
        adapted by this factor.
    sampling_2020_weights: List of weights to adapt the weight of the last 
        months before evaluation. This way countering the domain shift over time.
    label_column: Indicating what is the task. COVID_19 diagnosis prediction 
        with 'corona' or mortality prediciton with 'death_in_hospital'. 
        Indicates the column of the dataframe containing the label.
    test_months: A list containing the integers of the test months (the months, 
        the model is tested on).
    device: The device the torch models will be trained on (Logistic Regression
        and Self-Normalizing Neural Network). Can be either 'cpu' or, e.g., 'cuda:0'.
    
    Returns
    ---------
    AUC_val: The ROC AUC value calculated on the validation set. In case of no
        validation set, the AUC_val is None.
    model: The trained model.
    '''
    
    print('Start Training')
    
    if ((selected_model == 'SNN') or (selected_model == 'LOGREG')):
        AUC_val, model = train_loop(
            selected_model = selected_model, start_feature = start_feature, 
            keys = keys, hyperparamlist = hyperparamlist, k = k, 
            k_hyper = k_hyper, k_val = k_val, data = data, 
            train_indices = train_indices, criterion = criterion, 
            sampling_2019_weights = sampling_2019_weights, 
            sampling_2020_weights = sampling_2020_weights, 
            label_column = label_column, test_months = test_months, 
            trainloader = trainloader, valloader = valloader, device = device)
    
    elif ((selected_model == 'RF') or (selected_model == 'XGB')):
        if selected_model == 'RF':
            X = fulldataset.data[train_indices]
            y = fulldataset.labels[train_indices]
        else:
            X = data[data.keys()[start_feature:]].iloc[train_indices]
            y = data[label_column].iloc[train_indices]
        
        weights = get_weights(data = data, train_indices = train_indices, 
                              test_months = test_months, k = k, 
                              sampling_2020_weights = sampling_2020_weights, 
                              sampling_2019_weights = sampling_2019_weights)
              
        #normalize weight
        weights = weights * len(train_indices) / np.sum(weights)
        
        #balance
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        scale_pos = np.sum(weights[neg_indices]) / np.sum(weights[pos_indices])
        
        model = get_model(selected_model = selected_model, 
                          hyperparamlist = hyperparamlist, k_hyper = k_hyper, 
                          keys = keys, k = k, label_column = label_column, 
                          scale_pos = scale_pos)
                      
        model.fit(X, y, sample_weight = weights)

        if (val_indices is None):
            AUC_val = None
        else:
            if selected_model == 'RF':
                outputs = model.predict_proba(fulldataset.data[val_indices])[:,1]
            else:
                outputs = model.predict_proba(data[data.keys()[start_feature:]].iloc[val_indices])[:,1]
            AUC_val, _, _, _ = evaluate_test(prediction = outputs, target = fulldataset.labels[val_indices])

    elif ((selected_model == 'KNN') or (selected_model == 'SVM')):
        X = fulldataset.data[train_indices]
        y = fulldataset.labels[train_indices]
        model = get_model(selected_model = selected_model, 
                          hyperparamlist = hyperparamlist, k_hyper = k_hyper, 
                          keys = keys, k = k, label_column = label_column)

        model.fit(X, y)
        if (val_indices is None):
            AUC_val = None
        else:
            outputs = model.predict_proba(fulldataset.data[val_indices])
            AUC_val, _, _, _ = evaluate_test(outputs, fulldataset.labels[val_indices])
            
    if (AUC_val is not None):
        print(f'ROC AUC validation set: {AUC_val}')

    return AUC_val, model

def get_model(selected_model, hyperparamlist, keys, k = 0, k_hyper = 0,
              label_column = 'corona', scale_pos = 1):
    
    '''
    Get the initialized model with the respective hyperparameters.
    
    Parameters
    ----------
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    hyperparamlist: A list containing all possible hyperparameters. This list
        can be iterated in grid search.
    keys: The keys (names) of the hyperparameters stored in the dictionary.
    k: Counter of the test loop.
    k_hyper: Counter of the loop for the hyperparametersearch.
    label_column: Indicating what is the task. COVID_19 diagnosis prediction 
        with 'corona' or mortality prediciton with 'death_in_hospital'. 
        Indicates the column of the dataframe containing the label.
    scale_pos: Factor which adapts the scaling of the positive samples in XGB.
    
    Returns
    ---------
    init_model: The initialized model.
    '''
    
    if (selected_model == 'RF'):
        init_model = RandomForestClassifier(
            n_estimators = hyperparamlist[k_hyper][keys.index('n_estimators')],
            criterion = hyperparamlist[k_hyper][keys.index('criterion')],
            max_depth = hyperparamlist[k_hyper][keys.index('max_depth')],
            min_samples_split = hyperparamlist[k_hyper][keys.index('min_samples_split')],
            min_samples_leaf = hyperparamlist[k_hyper][keys.index('min_samples_leaf')],
            max_features = hyperparamlist[k_hyper][keys.index('max_features')],
            max_leaf_nodes = hyperparamlist[k_hyper][keys.index('max_leaf_nodes')],
            class_weight = hyperparamlist[k_hyper][keys.index('class_weight')],
            n_jobs = -1,
            random_state = k)
    elif (selected_model == 'XGB'):
        init_model = xgb.XGBClassifier(
            objective = hyperparamlist[k_hyper][keys.index('objective')],
            booster = hyperparamlist[k_hyper][keys.index('booster')],
            eta = hyperparamlist[k_hyper][keys.index('eta')],
            max_depth = hyperparamlist[k_hyper][keys.index('max_depth')],
            scale_pos_weight=scale_pos,
            grow_policy = hyperparamlist[k_hyper][keys.index('grow_policy')],
            eval_metric = hyperparamlist[k_hyper][keys.index('eval_metric')],
            n_jobs =  mp.cpu_count(),
            random_state = k,
            use_label_encoder = False)
    elif (selected_model == 'KNN'):
        init_model = KNeighborsClassifier(
            n_neighbors = hyperparamlist[k_hyper][keys.index('n_neighbors')],
            weights = hyperparamlist[k_hyper][keys.index('weights')],
            n_jobs = -1)
    elif (selected_model == 'SVM'):
        if (label_column == 'corona'):
            svm1 = svm.LinearSVC(dual = hyperparamlist[k_hyper][keys.index('dual')], 
                class_weight = hyperparamlist[k_hyper][keys.index('class_weight')]) 
            init_model = CalibratedClassifierCV(svm1)
        else:
            init_model = svm.SVC(probability = True, random_state = k)
            
    return init_model


def train_loop(selected_model, start_feature, keys, hyperparamlist,
                data, train_indices, trainloader, valloader, k = 0, 
                k_hyper = 0, k_val = 0, criterion = 'BCEWithLogitsLoss', 
                sampling_2019_weights = 1, sampling_2020_weights = [1,1,1,1], 
                label_column = 'corona', test_months = [11,12],  device = 'cpu'):
    
    '''
    Initializations and the training loop are called for Logistic Regression and
    Self-Normalizing Neural Network.
    The method returns the ROC AUC on the validation set and the trained model.
    
    Parameters
    ----------
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    start_feature: the keys of the dataframe are used for the model, starting 
        from the 'start_feature' index.
    keys: The keys (names) of the hyperparameters stored in the dictionary.
    hyperparamlist: A list containing all possible hyperparameters. This list
        can be iterated in grid search.
    data: Pandas dataframe containing the data set.
    train_indices: The indices of the training set.
    trainloader: Torch dataloader for training (required for LOGREG and SNN).
    valloader: Torch dataloader for validation (required for LOGREG and SNN).
    k: Counter of the test loop.
    k_hyper: Counter of the loop for the hyperparametersearch.
    k_val: Counter for the validation loop.
    criterion: Criterion to calculate the loss between prediction and target.
    sampling_2019_weights: The weight of the samples of the 2019 cohort is 
        adapted by this factor.
    sampling_2020_weights: List of weights to adapt the weight of the last 
        months before evaluation. This way countering the domain shift over time.
    label_column: Indicating what is the task. COVID_19 diagnosis prediction 
        with 'corona' or mortality prediciton with 'death_in_hospital'. 
        Indicates the column of the dataframe containing the label.
    test_months: A list containing the integers of the test months (the months, 
        the model is tested on).
    device: The device the torch models will be trained on (Logistic Regression
        and Self-Normalizing Neural Network). Can be either 'cpu' or, e.g., 'cuda:0'.
    
    Returns
    ---------
    AUC_val: The ROC AUC value calculated on the validation set. In case of no
        validation set, the AUC_val is None.
    model: The trained model.
    '''
    
    if selected_model == 'SNN':
        model = SNN(data.shape[1]-start_feature, 1, 
                    hyperparamlist[k_hyper][keys.index('intermediate_size')], 
                    hyperparamlist[k_hyper][keys.index('n_layers')], 
                    hyperparamlist[k_hyper][keys.index('dropout')])
    elif selected_model == 'LOGREG':
        model = logisticRegression(data.shape[1]-start_feature, 1)

    model = model.to(device)
    
    if (criterion == 'BCEWithLogitsLoss'):
        criterion = torch.nn.BCEWithLogitsLoss() 
    else:
        print('Criterion not defined. Using BCEWithLogitsLoss instead. \
              Add your loss to code.')
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = hyperparamlist[k_hyper][keys.index('lr')], 
        weight_decay = hyperparamlist[k_hyper][keys.index('weight_decay')])   
   
    AUC_val, model = run_epochs(
        selected_model = selected_model, trainloader = trainloader, 
        valloader = valloader, device = device, optimizer = optimizer,                              
        criterion = criterion, hyperparamlist = hyperparamlist, 
        k_hyper = k_hyper, keys = keys, model = model)
    
    return AUC_val, model
    

def run_epochs(selected_model, trainloader, valloader, optimizer, keys, model, 
               hyperparamlist, criterion = 'BCEWithLogitsLoss', k_hyper = 0,  
               device = 'cpu'):
    
    '''
    Run through the epochs until early stopping criterion or n_epochs is met.
    Training and validation for Logistic Regression and Self-Normalizing Neural 
    Network. 
    
    Parameters
    ----------
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    trainloader: Torch dataloader for training.
    valloader: Torch dataloader for validation.
    optimizer: Torch optimizer to optimize the objective.
    keys: The keys (names) of the hyperparameters stored in the dictionary.
    model: The initialized model.
    hyperparamlist: A list containing all possible hyperparameters. This list
        can be iterated in grid search.
    criterion: Criterion to calculate the loss between prediction and target.
    k_hyper: Counter of the loop for the hyperparametersearch.
    device: The device the torch models will be trained on (Logistic Regression
        and Self-Normalizing Neural Network). Can be either 'cpu' or, e.g., 
        'cuda:0'.
    
    Returns
    ---------
    AUC_val: The ROC AUC value calculated on the validation set. In case of no
        validation set, the AUC_val is None.
    model_store: The trained model.
    '''

    n_epochs, loss, trainloss, valloss, trainloss_store, valloss_store, \
        best_val_loss, stop, val_counter, n_val_stops = \
            initialize_run_epochs(hyperparamlist = hyperparamlist, 
                                  k_hyper = k_hyper, keys = keys)
    
    for epoch in range(n_epochs):
        if(stop == 1):
            break
        
        #training
        for index, mydata in enumerate(trainloader):
            if stop == 1:
                break

            inputs, labels = mydata
            outputs = model.train()(inputs.to(device))
            loss = criterion(outputs[:, 0], labels.to(device))
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
            optimizer.zero_grad()
    
        #validation
        with torch.no_grad():
            for index, mydata in enumerate(valloader):
                optimizer.zero_grad()
                inputs, labels = mydata
                outputs = model.eval()(inputs.to(device))
                loss = criterion(outputs[:, 0], labels.to(device))
                valloss += loss.item()
            
            #early stopping
            if (valloss < best_val_loss):
                best_val_loss = valloss
                val_counter = 0
                model_store = copy.deepcopy(model)
            else:
                val_counter+=1
                if(val_counter > n_val_stops):
                    stop = 1                    
            
        valloss_store.append(valloss)
        valloss = 0
        trainloss_store.append(trainloss)
        trainloss = 0
      
        print('epoch {}, train_loss {}'.format(epoch, trainloss_store[-1]))
        print('epoch {}, val_loss {}'.format(epoch, valloss_store[-1]))
    
    #validation
    outputs, labels = test_model(testloader = valloader, 
                                 final_model = model_store, 
                                 selected_model = selected_model, 
                                 device = device)
    AUC_val, _, _,_ = evaluate_test(prediction = outputs, target = labels)

    return AUC_val, model_store


def initialize_run_epochs(hyperparamlist, keys, k_hyper = 0):
    
    '''
    Initialization of variables required for monitoring the training progress
    
    Parameters
    ----------
    hyperparamlist: A list containing all possible hyperparameters. This list
        can be iterated in grid search.
    keys: The keys (names) of the hyperparameters stored in the dictionary.
    k_hyper: Counter of the loop for the hyperparametersearch.
        
    Returns
    n_epochs: Maximum number of epochs.
    loss: Zero initialized variable. Will contain the loss.
    trainloss: Zero initialized variable. Will contain the trainingloss.
    valloss: Zero initialized variable. Will contain the validationloss.
    trainloss_store: Empty list, which will be appended with the training loss
        at each epoch.
    valloss_store:Empty list, which will be appended with the validation loss
        at each epoch.
    best_val_loss: Initialized with infinite. Will contains the so far best 
        loss. Used to check, whether the loss is still decreasing.
    stop: Zero initialized binary variable, it is set to 1 in case early 
        stopping criterion is met.
    val_counter: Zero initialized variable. In case the validation loss is not 
        decreasing any more, the val_counter will start to count until 
        'n_val_stops'. Then the early stopping criterion is met.
    n_val_stops: Zero initialized variable. The validation loss can exceed the 
        so far best loss for 'n_val_stops' until the training procedure is 
        terminated (early stopping).
    ---------
    
    '''
    
    n_epochs = 1000
    loss = 0
    trainloss = 0
    valloss = 0
    trainloss_store = []
    valloss_store = []
    best_val_loss = np.inf
    stop = 0
    val_counter = 0
    n_val_stops = hyperparamlist[k_hyper][keys.index('n_val_stops')]
    
    return n_epochs, loss, trainloss, valloss, trainloss_store, valloss_store, \
        best_val_loss, stop, val_counter, n_val_stops
    
    