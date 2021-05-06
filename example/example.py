#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at
        
This is an example application including:
    (.) Generation of a random data set or loading of your own data set
    (.) Preprocessing of the data set
    (.) Training of a COVID-19 or mortality prediction model
    (.) Hyperparameter search or prospective evaluation
    
The configurations can be set in 'config.py'.

IMPORTANT: Note that the randomly generated data is just toy data and models 
based on these data are not suitable for in-hospital COVID-19 diagnosis or 
mortality prediction.
Load the data from your institution, train the models and properly evaluate the 
model to avoid unexpected model behavior.
"""

import pandas as pd
import numpy as np
import copy as copy
from config import *
import sys
sys.path.append('../')
from preprocessing import generate_preprocess_dataset, initialize, \
        initialize_hyperparams, initialize_validation, split_dataset, impute, \
        normalize, get_dataset_dataloader
from training import train_model
from evaluate_model import test, evaluate
from store_load_model import store_model

''' 
Generate random data file for the toy example and preprocess the data set.
The random data can be replaced by your COVID-19 data
empty features contain NaN
'''

data, start_feature, len_coronaset = generate_preprocess_dataset(
    do_generate_random_data = do_generate_random_data,
    len_coronaset = len_coronaset,
    data_path = data_path,
    min_n_entries = min_n_entries,
    n_lab_entries = n_lab_entries,
    label_column = label_column,
    use_negatives_dataset = use_negatives_dataset)

'''
Initializations, prepare training
'''

AUC_val, target_stacked, prediction_stacked, best_hyperparams, final_models = \
    initialize(n_folds = len(test_months), n_val_folds = n_val_folds)
    
'''
The model is trained and evaluated with following loops:
    (.) Loop for test folds/test months
    (.) Loop for hyperparameters in case of hyperparamsearch,
        in case of no hyperparamsearch this loop is run only once
    (.) Loop for validation folds
'''
               
'''Loop for test folds/test months'''
for k in range(0,max([1,len(test_months)])):

    '''Initialize hyperparamsearch'''
    hyperparams, AUC_hyper, AUC_hyper_max, keys, hyperparamlist, \
    n_hyper_combinations = initialize_hyperparams(selected_model = selected_model, 
                                                  label_column = label_column)
    
    '''Loop for hyperparamsearch'''
    for k_hyper in range(0, n_hyper_combinations):
        
        '''Initialize validation'''
        AUC_val_max = initialize_validation()
        
        '''Loop for validation folds'''
        for k_val in range(0,n_val_folds):
                        
            '''Split the dataset into train/val/testset'''
            train_indices, val_indices, test_indices = \
            split_dataset(data = data, test_months = test_months, 
                          n_hyper_combinations = n_hyper_combinations, k = k, 
                          selected_model = selected_model)
                                               
            '''Imputation with median'''
            data = impute(data = data, train_indices = train_indices, 
                          start_feature = start_feature)
            
            '''Normalization (Z-score normalization)'''
            data = normalize(data = data, train_indices = train_indices, 
                             start_feature = start_feature)

            '''Make dataset and dataloader    '''
            trainloader, valloader, testloader, n_pos, n_neg, fulldataset = \
                get_dataset_dataloader(data = data, start_feature = start_feature,
                                       label_column = label_column, 
                                       train_indices = train_indices, 
                                       val_indices = val_indices,
                                       test_indices = test_indices, 
                                       test_months = test_months, k = k, 
                                       sampling_2019_weights = sampling_2019_weights,
                                       sampling_2020_weights = sampling_2020_weights,
                                       batch_size = batch_size)
            
            '''Train'''
            AUC_val[k, k_val], trained_model = train_model(
                selected_model = selected_model, start_feature = start_feature, 
                keys = keys, hyperparamlist = hyperparamlist, k = k, 
                k_hyper = k_hyper, k_val = k_val, data = data, 
                train_indices = train_indices, val_indices = val_indices, 
                criterion = criterion, sampling_2019_weights = sampling_2019_weights, 
                sampling_2020_weights = sampling_2020_weights, 
                label_column = label_column, test_months = test_months, 
                fulldataset = fulldataset, trainloader = trainloader, 
                valloader = valloader, device = device)
            
            if ((val_indices is not None) and (AUC_val[k, k_val] > AUC_val_max)):
                AUC_val_max = AUC_val[k, k_val]
                model_val_intermediate = copy.deepcopy(trained_model)
            elif (val_indices is None):
                model_val_intermediate = copy.deepcopy(trained_model)
            
            
        if (val_indices is not None):
            AUC_hyper[k_hyper] = np.mean(AUC_val[k, k_val])
        '''
        Check if this is the best model on validation set, if yes, then store it
        '''
        if ((AUC_hyper[k_hyper] > AUC_hyper_max) or (val_indices is None)):   
            '''
            In case of multiple validation splits take the last trained model 
            with this hyperparamsetting
            '''
            final_model = copy.deepcopy(model_val_intermediate)
            params_selected = hyperparamlist[k_hyper]
            if (val_indices is not None):
                AUC_hyper_max = np.mean(AUC_val[k, k_val])           
    
    if (hyperparamsearch):   
        best_hyperparams.append(params_selected)
        final_models.append(final_model)
    
    '''Evaluation on test set'''
    target_stacked, prediction_stacked = test(
        target_stacked = target_stacked, prediction_stacked = prediction_stacked, 
        selected_model = selected_model, testloader = testloader, 
        final_model = final_model, data = data, fulldataset = fulldataset, 
        test_indices = test_indices, start_feature = start_feature, 
        device = device)
  
'''
Evaluate
Stack the predictions and targets of all test months and evaluate these.
The performance of the models on the test months is evaluated on the basis of 
ROCAUC, PRAUC with their 95% confidence intervals, determined by bootstrapping.
'''

ROCAUC, CI_ROCAUC, PRAUC, CI_PRAUC = evaluate(
    selected_model = selected_model, prediction_stacked = prediction_stacked,
    target_stacked = target_stacked, hyperparamsearch = hyperparamsearch, 
    best_hyperparams = best_hyperparams, n_folds = max([1,len(test_months)]))

'''
Store trained model.
'''

store_model(store_trained_model = store_trained_model, 
            selected_model = selected_model, label_column = label_column, 
            final_model = final_model, test_months = test_months, 
            path_store_model = path_store_model)



