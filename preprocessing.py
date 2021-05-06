#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at
        
Preprocessing of the data set.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import itertools
from hyperparams import *
from generate_random_data import generate_random_data
from architectures import myDataset



def generate_preprocess_dataset(do_generate_random_data, len_coronaset,
                                data_path, min_n_entries, n_lab_entries = None,    
                                label_column = 'corona', 
                                use_negatives_dataset = True):
    
    '''
    Generate a random dataset as a toy example and preprocess the data set.
    
    Parameters
    ----------
    do_generate_random_data: A boolean, which indicates whether the dataset has 
        to be generated or is loaded from a file. 
    len_coronaset: The number of samples for the 2020 cohort. The 2019 cohort is 
        stacked after the 2020 cohort in the dataframe.
    data_path: The path where the raw data structure is stored at and where the
        generated data will be stored at.
    min_n_entries: Each sample requires a minimum of 'min_n_entries'.
        Otherwise the sample is discarded.
    n_lab_entries: Random samples will be generated n_lab_entries times and 
        entered at a random feature and sample to random posititions.    
    label_column: Indicating what is the task. COVID_19 diagnosis prediction 
        with 'corona' or mortality prediciton with 'death_in_hospital'. 
        Indicates the column of the dataframe containing the label.
    use_negatives_dataset: Boolean. Whether the 2019 cohort is used or discarded.
    
    Returns
    ---------
    data: Pandas dataframe containing the data set.
    start_feature: The keys of the dataframe are used for the model, starting 
        from the 'start_feature' index. 
    len_coronaset: The number of samples for the 2020 cohort. The 2019 cohort is 
        stacked after the 2020 cohort in the dataframe.
    '''
    
    if (do_generate_random_data):
        generate_random_data(n_lab_entries = n_lab_entries, len_coronaset = len_coronaset,
                         data_path = data_path)

    #load random data or your data
    data = pd.read_feather(data_path + '/data_random.feather')
    
    print('Start Preprocessing')
    #the start_feature is the index of the first input feature for the model
    #the features before start_feature are meta information
    start_feature = np.where(data.keys() == 'type')[0][0]
    
    #drop 2019 set if not used
    if (not use_negatives_dataset):
        data = data.loc[0:len_coronaset]
    
    if (label_column == 'death_in_hospital'):
        #drop COVID-19 negative samples
        data_with_corona = data.iloc[0:len_coronaset]
        data = data_with_corona.drop(index = data_with_corona[data_with_corona['corona']==0].index).reset_index(drop=True)
        use_negatives_dataset = False     
    
    #select the features the model is based on and drop samples with too few entries
    data, selected_columns, len_coronaset = select_columns(
        data = data, start_feature = start_feature, len_coronaset = len_coronaset,
        min_n_features = min_n_features, min_n_entries = min_n_entries, 
        use_negatives_dataset = use_negatives_dataset)
    
    #add a feature ('MISSING_') indicating whether the feature is measured
    data = add_missing_features(data=data, start_feature=start_feature)
    
    return data, start_feature, len_coronaset


def select_columns(data, start_feature, len_coronaset, min_n_features, 
                   min_n_entries, use_negatives_dataset = True):
    
    '''
    Select the most frequent columns/features, which are used for the model.
    The other columns are droped from the data set and not used for model training.
    
    Parameters
    ----------
    data: Pandas dataframe containing the data set.
    start_feature: The keys of the dataframe are used for the model, starting 
        from the 'start_feature' index.
    len_coronaset: The number of samples for the 2020 cohort. The 2019 cohort is 
        supposed to be stacked after the 2020 cohort in the data dataframe.
    min_n_features: The most frequently present features are selected for the 
        models. A minimum of 'min_n_features' features is selected. In case 
        features are measured equally often, the number of the minimum features might be exceeded slightly. 
    min_n_entries: Each sample requires a minimum of 'min_n_entries'.
        Otherwise the sample is discarded.
    use_negatives_dataset: Whether the 2019 cohort is used. In case it is used, 
        the features for the COVID-19 diagnosis prediction models are selected  
        on the basis of the 2019 cohort. Otherwise the most frequent features are
        determined on the 2020 cohort.
        
    Returns
    ---------
    data: Pandas data frame with the selected features and samples.
    selected_columns: the keys (strings) of the selected columns.
    len_coronaset: The number of the selected samples of the 2020 cohort.
    '''

    np.random.seed(0)
    torch.manual_seed(0)
    
    mycolumns = data.keys()
    n_nans = np.zeros((data.shape[1] - start_feature))
    sd = np.zeros((data.shape[1] - start_feature))
    for i in range(start_feature, data.shape[1]):
        if (use_negatives_dataset):
            #select the features on basis of negatives dataset
            n_nans[i-start_feature] = np.sum(np.isnan(
                data.loc[len_coronaset:, mycolumns[i]].values))
            myvalues = data.loc[len_coronaset:, mycolumns[i]].values
            sd[i-start_feature] = np.nanstd(myvalues[np.isfinite(myvalues)])
        else:
            #select the features on basis of coronaset
            n_nans[i - start_feature] = np.sum(np.isnan(
                data[mycolumns[i]].values))
            myvalues = data[mycolumns[i]].values
            sd[i - start_feature] = np.nanstd(myvalues[np.isfinite(myvalues)])
    sd[np.isnan(sd)] = 0
    
    #the larger 'n_iterator', the more exact the min_n_features can be achieved
    n_iterator = 100000 
    for i in range(n_iterator):
        #keep these columns
        indices = np.where((n_nans < (len(data) * i / n_iterator)) & (sd != 0))[0] 
        if len(indices) >= min_n_features:
            break
    
    selected_columns = [mycolumns[index + start_feature] for index in indices]
    selected_columns = (list(data.keys()[0:start_feature].values) 
                        + selected_columns[:])
    
    #the dataset only contains the selected columns
    data = data[selected_columns].copy()

    #get number of rows with less than xx entries
    n_entries = np.sum(np.isfinite(np.array(data.values[:, start_feature:], 
                                            dtype=float)), axis=1)
    
    #drop the entries with less than xx features
    dropindices = np.where(n_entries < min_n_entries)[0]
    data = data.drop(index = dropindices).reset_index(drop=True)
    
    #correct len_coronaset by the droped samples
    n_droped_samples_in_coronaset = np.sum(dropindices < len_coronaset)
    len_coronaset -= n_droped_samples_in_coronaset
    
    return data, selected_columns, len_coronaset


def add_missing_features(data, start_feature):
    '''
    Add the feature 'MISSING_...' indicating whether a feature is missing or 
    measured. New features are created here. MISSING_ are binary indicator features.
    
    Parameters
    ----------
    data: Pandas dataframe containing the data set.
    start_feature: the keys of the dataframe are used for the model, starting 
        from the 'start_feature' index.
        
    Returns
    ---------
    data: Pandas data frame with the feature columns increased by the 'MISSING_' 
        features.
    '''
    #get the length of the features
    end_norm_features = len(data.keys())

    missing_features = np.zeros((len(data), len(data.keys()[start_feature:])))
    for i, key in enumerate(data.keys()[start_feature:]):
        #non-measured features are indicated with a 1
        missing_features[:,i] = np.isnan(data[key])*1
        
    newcolumns = []
    for i, mykey in enumerate(data.keys()[start_feature:]):
        newcolumns.append('MISSING_' + mykey)
      
    #append the dataframe with the 'MISSING' features
    data[newcolumns] = pd.DataFrame(missing_features)
    data[data.keys()[start_feature:end_norm_features]] = \
        data[data.keys()[start_feature:end_norm_features]].astype(np.float32)
    
    return data


def impute(data, train_indices, start_feature):
    
    '''
    Missing features (NaN) are imputated with median as calculated from the 
    available training samples.
    
    Parameters
    ----------
    data: Pandas dataframe containing the data set.
    train_indices: The indices of the training set. These are required to 
        calculate the training set based median.
    start_feature: the keys of the dataframe are used for the model, starting 
        from the 'start_feature' index.
        
    Returns
    ---------
    data: Pandas data frame with the nan values imputed by the training set 
        based median. Feature columns are increased by the 'MISSING_' features.
    '''
    
    end_norm_features = start_feature+int(len(data.iloc[0, start_feature:])/2)
    median = np.nanmedian(np.array(
            data.iloc[train_indices, start_feature:end_norm_features]),axis=0)
    for i, key in enumerate(data.keys()[start_feature:end_norm_features]):
        if np.isnan(median[i]):
            data[key] = data[key].fillna(0) 
        else:
            data[key] = data[key].fillna(median[i]) 
    return data


def normalize(data, train_indices, start_feature):
    
    '''
    Z-Score Normalization
    Normalize to 0 mean and standard deviation 1 (on the basis of training data)
    
    Parameters
    ----------
    data: Pandas dataframe containing the data set.
    train_indices: The indices of the training set. These are required to 
        calculate the training set based mean and standard deviation required 
        for Z-Score Normalization.
    start_feature: the keys of the dataframe are used for the model, starting 
        from the 'start_feature' index.
        
    Returns
    ---------
    data_copy: Pandas data frame with the normalized feature values.
    '''
    
    end_norm_features = (start_feature 
                         + int(len(data.iloc[0, start_feature:]) / 2))
    #get mean and standard deviation from training set
    mymean = np.nanmean(np.array(data.iloc[train_indices, 
                         start_feature:end_norm_features]),axis=0)
    mysd = np.nanstd(np.array(data.iloc[train_indices, 
                         start_feature:end_norm_features]),axis=0)
   
    data_copy = data.copy() 
    zero_sd_counter = 0
    for i in range(mymean.shape[0]):
        #no division by SD if it is 0
        if mysd[i] == 0:
            data_copy[data.keys()[start_feature + i:start_feature + i + 1]] = \
                (data[data.keys()[start_feature + i:start_feature + i + 1]] 
                 - mymean[i])
            zero_sd_counter += 1
        else:
            data_copy[data.keys()[start_feature + i:start_feature + i + 1]] = \
                (data[data.keys()[start_feature + i:start_feature + i + 1]] 
                 - mymean[i]) / mysd[i]

    return data_copy


def split_dataset(data, selected_model, test_months = [11, 12], 
                  n_hyper_combinations = 1, k = 0, k_val = 0):
    
    '''
    Split the dataset into train/val/testset.
    The testset is given by the test_months.
    A validation set is not required for prospective evaluation for all models 
    but SNN and LOGREG. SNN and LOGREG need a validation set in any case for 
    early stopping. The validation set is randomly drawn from the train_val_set 
    and it must not contain samples from any test month. The validation set
    does not contain samples from the 2019 cohort.
    The training set are the remaining samples.
    
    Parameters
    ----------
    data: Pandas dataframe containing the data set.
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    test_months: A list containing the integers of the test months (the months, 
        the model is tested on).
    n_hyper_combinations: This is required to check, whether validation will be 
        done for hyperparameterselection. If it is 1, hyperparametersearch will 
        not be conducted.
    k: Counter of current test fold.
    k_val: Counter of current validation fold.
    
    
    Returns
    ---------
    train_indices: Indices of the samples of the training set.
    val_indices: Indices of the samples of the validation set (might be None if 
        no validation set is used, e.g., for prospective evaluation).
    test_indices: Indices of the samples of the test set.
    '''
    
    np.random.seed(k_val)
    torch.manual_seed(k_val)
         
    if test_months == []:
        test_months = [np.inf]
        test_indices = None
        train_val_indices = data[data['Month'] < np.inf].index
    else:
        test_indices = data[data['Month'] == (test_months[k])].index
        train_val_indices = data[data['Month'] < test_months[k]].index
        
    if ((n_hyper_combinations <= 1) and not ((selected_model == 'SNN') or 
                                           (selected_model == 'LOGREG'))):
        #no hyperparam search, 
        #we can use the train and validation set for training the model
        train_indices = train_val_indices
        #no validation set
        val_indices = None
        
    else:
        #hyperparam search
        #validation set to determine the hyperparameters
        #the validation must not be done on samples from the test set
        train_val_indices = data[(data['Month'] < min(test_months)) & (data['Month'] > 1)].index
        val_indices = np.random.choice(train_val_indices, 
                                       int(len(train_val_indices) * 0.20))
        train_val_indices = data[data['Month'] < test_months[k]].index
        train_indices = [i for i in train_val_indices if i not in val_indices]
    
    return train_indices, val_indices, test_indices


def get_dataset_dataloader(data, start_feature, train_indices, test_indices, 
                           val_indices = None, label_column = 'corona', 
                           test_months = [11,12], k = 0, 
                           sampling_2019_weights = 1, 
                           sampling_2020_weights = [1,1,1,1], batch_size = 256):
    
    '''
    Get datasets and dataloaders (required for the Pytorch models: Logistic 
    Regression and Self-Normalizing Neural Network).
    
    Parameters
    ----------
    data: Pandas dataframe containing the data set.
    start_feature: the keys of the dataframe are used for the model, starting 
        from the 'start_feature' index.
    train_indices: The indices of the training set. 
    test_indices: The indices of the training set. 
    val_indices: The indices of the validation set. A validation set is required 
        for Logistic Regression and Self-Normalizing Neural Network. 
        For the other models, a validation set is optional (but also required 
        for hyperparameter search).
    label_column: Indicating what is the task. COVID_19 diagnosis prediction 
        with 'corona' or mortality prediciton with 'death_in_hospital'. 
        Indicates the column of the dataframe containing the label.
    test_months: A list containing the integers of the test months (the months, 
        the model is tested on).
    k: Counter of current test fold.
    sampling_2019_weights: The weight of the samples of the 2019 cohort is 
        adapted by this factor.
    sampling_2020_weights: List of weights to adapt the weight of the last 
        months before evaluation. This way countering the domain shift over time.
    batch_size: Batchsize for dataloaders. Only required for Pytorch models 
        (Logistic Regression and Self_Normalizing Neural Network).
        
    Returns
    ---------
    trainloader: Torch dataloader for training.
    valloader: Torch dataloader for validation.
    testloader: Torch dataloader for testing.
    n_pos: Number of positive samples in training set.
    n_neg: Number of negative samples in training set.
    fulldataset: Torch dataset containing all data.
    '''
    
    fulldataset = myDataset(data = data, start_feature = start_feature, 
                            label_column = label_column)
            
    #subsets
    trainset = Subset(fulldataset, np.array(train_indices).astype(int))
    if (val_indices is None):
        valset = None
    else:
        valset = Subset(fulldataset, val_indices.astype(int))
    if (test_indices is None):
        testset = None
    else:
        testset = Subset(fulldataset, test_indices.astype(int))
    
    n_pos, n_neg = count_positives_negatives(fulldataset = fulldataset, 
                                             train_indices = train_indices)
       
    weights = get_weights(data = data, train_indices = train_indices, 
                          test_months = test_months, k = k, 
                          sampling_2020_weights = sampling_2020_weights, 
                          sampling_2019_weights = sampling_2019_weights)
    
    #normalize weight
    weights = weights / len(train_indices)
    
    #balance positive and negative class
    weights = balance(y = fulldataset.labels[train_indices], weights = weights)
        
    trainsampler = torch.utils.data.WeightedRandomSampler(
        torch.FloatTensor(weights), len(train_indices))
    
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = False, 
                             sampler = trainsampler, batch_sampler = None, 
                             num_workers = 0, collate_fn = None,
                             pin_memory = False, drop_last = False)
    
    if (val_indices is None):
        valloader = None
    else:
        valloader = DataLoader(valset, batch_size = batch_size, shuffle = False, 
                               sampler = None, batch_sampler = None, 
                               num_workers = 0, collate_fn = None, 
                               pin_memory = False, drop_last = False)
    
    if test_indices is  None:
        testloader = None
    else:
        testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, 
                                sampler = None, batch_sampler = None, 
                                num_workers = 0, collate_fn = None, 
                                pin_memory = False, drop_last = False)

    return trainloader, valloader, testloader, n_pos, n_neg, fulldataset


def count_positives_negatives(fulldataset, train_indices):
    
    '''
    Get the number of positive and negative samples 
    
    Parameters
    ----------
    fulldataset: Torch dataset containing all data.
    train_indices: Indices of the training samples.
    
    Returns
    ---------
    n_pos: Number of positive samples in training set.
    n_neg: Number of negative samples in training set.
    '''

    train_labels = fulldataset.labels[train_indices]
    n_pos = torch.sum(train_labels == 1)
    n_neg = len(train_labels) - n_pos
    
    return n_pos, n_neg


def get_weights(data, train_indices, test_months = [11,12], k = 0, 
                sampling_2020_weights = [1,1,1,1], sampling_2019_weights = 1):
    
    '''
    Get the weights in dependence of sample recency.
    
    Parameters
    ----------
    data: Pandas dataframe containing the data set.
    train_indices: The indices of the training set.
    test_months: A list containing the integers of the test months (the months, 
        the model is tested on).
    k: Counter of current test fold.
    sampling_2020_weights: list of weights to adapt the weight of the last 
        months before evaluation. This way countering the domain shift over time.
    sampling_2019_weights: the weight of the samples of the 2019 cohort is 
        adapted by this factor.
        
    Returns
    ---------
    weights: An array containing the weight of each sample.
    '''
    
    if (len(test_months)<=k):
        test_months = np.zeros((k+1))
        test_months[k] = max(data['Month'])+1
    
    month_before = data.iloc[train_indices][data.iloc[train_indices]['Month']
                                            == (test_months[k] - 1)].index
    month_2_before = data.iloc[train_indices][data.iloc[train_indices]['Month']
                                              ==( test_months[k] - 2)].index
    month_3_before = data.iloc[train_indices][data.iloc[train_indices]['Month']
                                              == (test_months[k] - 3)].index
    month_4_before = data.iloc[train_indices][data.iloc[train_indices]['Month']
                                              == (test_months[k] - 4)].index
    month_2019 = data.iloc[train_indices][data.iloc[train_indices]['Month']
                                              <= 1].index
    weights = np.ones(len(train_indices))
    ind_before = np.where(np.in1d(train_indices, month_before))
    ind_2_before = np.where(np.in1d(train_indices, month_2_before))
    ind_3_before = np.where(np.in1d(train_indices, month_3_before))
    ind_4_before = np.where(np.in1d(train_indices, month_4_before))
    ind_2019 = np.where(np.in1d(train_indices, month_2019))

    weights[ind_before] = weights[ind_before] * sampling_2020_weights[-1]
    weights[ind_2_before] = weights[ind_2_before] * sampling_2020_weights[-2]
    weights[ind_3_before] = weights[ind_3_before] * sampling_2020_weights[-3]
    weights[ind_4_before] = weights[ind_4_before] * sampling_2020_weights[-4]
    
    weights[ind_2019] = weights[ind_2019] * sampling_2019_weights
    
    return weights


def balance(y, weights):
    
    '''
    Balance the weights of positive and negative class
    
    Parameters
    ----------
    y: The targets (labels) of the training set.
    weights: An array containing the weight of each sample.
    
    Returns
    ---------
    weights: An array containing the balanced weight of each sample.
    '''

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    weights[pos_indices] = weights[pos_indices] / (np.sum(weights[pos_indices]))
    weights[neg_indices] = weights[neg_indices] / (np.sum(weights[neg_indices]))
    
    return weights

def initialize(n_folds, n_val_folds):
    
    '''
    Initialize variables for monitoring the training, validation and testing.
    
    Parameters
    ----------
    n_folds: Number of test folds/test months.
    n_val_folds: Number of validation folds.
    
    Returns
    ---------
    AUC_val: Initialized with zeros with dimensions test folds and validation 
        folds. The ROC AUC on the validation set will be written to this array.
    target_stacked: Empty list, which will be appended by the targets (labels) 
        on the test sets/test months.
    prediction_stacked: Empty list, which will be appended by the predictions 
        on the test sets/test months.
    best_hyperparams: Empty list, which will be filled with the selected 
        hyperparameters.
    final_models: Empty list, which will be filled with the selected trained 
        models.
    '''
    
    AUC_val = np.zeros((max([1,n_folds]), max([1,n_val_folds])))
    target_stacked = []
    prediction_stacked = []
    best_hyperparams = []
    final_models = []
    
    return AUC_val, target_stacked, prediction_stacked, best_hyperparams, \
            final_models
  

def initialize_hyperparams(selected_model, label_column = 'corona', k = 0):
    
    '''
    Initialize hyperparam search.
    
    Parameters
    ----------
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    label_column: Indicating what is the task. COVID_19 diagnosis prediction 
        with 'corona' or mortality prediciton with 'death_in_hospital'. 
        Indicates the column of the dataframe containing the label.
    k: Counter of current test fold.
    
    Returns
    ---------
    hyperparams: A dictionary of the hyperparameters.
    AUC_hyper: The ROC AUC on the validation set for each hyperparameter setting.
    AUC_hyper_max: Zero initialized variable to track the maximum hyperparameter 
        ROC AUC.
    keys: The keys (names) of the hyperparameters stored in the dictionary.
    hyperparamlist: A list containing all possible hyperparameters. This list
        can be iterated in grid search.
    n_hyper_combinations: The length of the hyperparamlist. The number of all
        possible combinations of the different hyperparameters.        
    '''
    
    np.random.seed(k)
    
    AUC_hyper_max = 0
    hyperparams = eval(selected_model + '_params')
    if ( (selected_model == 'SVM') and (label_column == 'corona') ):
        hyperparams = eval(selected_model + '_linear_params')
    keys = list(hyperparams.keys())
    hyperparamlist = list(itertools.product(*hyperparams.values()))
    n_hyper_combinations = len(hyperparamlist)
    AUC_hyper = np.zeros((n_hyper_combinations))
    
    return hyperparams, AUC_hyper, AUC_hyper_max, keys, hyperparamlist, \
            n_hyper_combinations
     

def initialize_validation():
    
    '''
    Initialize variables for validation.
    
    Parameters
    ----------
    None
    
    Returns
    ---------
    AUC_val_max: Zero initialized variable. Used for tracking the maximal ROC AUC 
        value in multiple validation folds.
    '''
    
    AUC_val_max = 0
    
    return AUC_val_max