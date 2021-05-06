#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at
        
Evaluate the stacked predictions and targets.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
import torch

def test(target_stacked, prediction_stacked, selected_model, final_model, data, 
         testloader = None, fulldataset = None, test_indices = None, 
         start_feature = None, device = 'cpu'):
    
    '''
    Test the final model.
    
    Parameters
    ----------
    target_stacked: The targets (true labels) stacked for all test folds/months.
    prediction_stacked: The predictions stacked for all test folds/months.
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    final_model: Trained torch model (e.g., Logistic Regression model or 
        Self-Normalizing Neural Network model).
    data: Pandas dataframe containing the data set.
    testloader: Torch dataloader for testing (only required for LOGREG and SNN).
    fulldataset: Torch dataset containing all data (required for all models but 
        NOT for 'LOGREG' and 'SNN').
    test_indices: Indices of the samples of the test set (required for all 
        models but NOT for 'LOGREG' and 'SNN').
    start_feature: the keys of the dataframe are used for the model, starting 
        from the 'start_feature' index (required only for 'XGB').
    device: The device the torch models will be trained on (Logistic Regression
        and Self-Normalizing Neural Network). Can be either 'cpu' or, e.g., 'cuda:0'.
    
    Returns
    ---------
    
    target_stacked: The targets (true labels) stacked for all test folds/months.
    prediction_stacked: The predictions stacked for all test folds/months.
    '''
    
    if ((testloader == None) and (test_indices == None)):
       print('No Testing')
       target_stacked = None
       prediction_stacked = None
    else:
        if (selected_model in ['SNN', 'LOGREG']):            
            myoutputs, mylabels = test_model(testloader = testloader, 
                                             final_model = final_model, 
                                             selected_model = selected_model, 
                                             device = device)
            target_stacked.append(mylabels)
            prediction_stacked.append(myoutputs)
        else:
            if (selected_model == 'XGB'):
                myoutputs = final_model.predict_proba(data[data.keys()[start_feature:]].iloc[test_indices])
            else:
                myoutputs = final_model.predict_proba(fulldataset.data[test_indices])
            target_stacked.append(fulldataset.labels[test_indices])
            prediction_stacked.append(myoutputs[:, 1])

    return target_stacked, prediction_stacked

def test_model(testloader, final_model, selected_model, device = 'cpu'):
    
    '''
    Return the predictions and labels for the testset for Logistic Regression
    and Self-Normalizing Neural Network.
    
    Parameters
    ----------
    trainloader: Torch dataloader for training.
    final_model: Trained torch model (e.g., Logistic Regression model or 
        Self-Normalizing Neural Network model).
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    device: The device the torch models will be trained on (Logistic Regression
        and Self-Normalizing Neural Network). Can be either 'cpu' or, e.g., 'cuda:0'.
    
    Returns
    ---------
    outputs: The predictions on the testset.
    labels: The targets (true labels) on the testset.
    '''
    
    list_outputs = []
    list_labels = []
    with torch.no_grad():
        valloss = 0
        for index, mydata in enumerate(testloader):
            inputs, labels = mydata
            outputs = final_model.eval()(inputs.to(device))
            list_outputs.extend(outputs.detach().cpu())
            list_labels.extend(labels.detach().cpu())
    
    outputs = torch.stack(list_outputs)
    labels = torch.stack(list_labels)
    
    return outputs, labels


def evaluate_test(prediction, target):
    
    '''
    Calculate the ROC AUC and PR AUC and their 95% confidence interval determined
    with bootstrapping n_bootstraps times.
    
    Parameters
    ----------
    prediction: The predictions by a model.
    target: The targets (true labels).
    
    Returns
    ---------
    ROCAUC: Area under the receiver operating curve.
    CI_ROCAUC: 95% confidence interval, determined by bootstrapping.
    PRAUC: Area under the precision recall curve.
    CI_PRAUC: 95% confidence interval, determined by bootstrapping.
    '''
    
    n_bootstraps = 1000
    ROCAUC_bootstrapped_scores = []
    PRAUC_bootstrapped_scores = []
    for i in range(n_bootstraps): 
        indices = np.random.randint(0, len(prediction), len(prediction))
        if len(np.unique(target[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(target[indices], prediction[indices])
        ROCAUC_bootstrapped_scores.append(score)
        precision, recall, thresholds = \
            precision_recall_curve(target[indices], prediction[indices])
        score = auc(recall, precision)
        PRAUC_bootstrapped_scores.append(score)
    ROCAUC_sorted_scores = np.array(ROCAUC_bootstrapped_scores)
    ROCAUC_sorted_scores.sort()
    CI_ROCAUC = [ROCAUC_sorted_scores[int(0.025 * len(ROCAUC_sorted_scores))], 
                 ROCAUC_sorted_scores[int(0.975 * len(ROCAUC_sorted_scores))]]
    ROCAUC = np.mean(ROCAUC_sorted_scores)
    
    PRAUC_sorted_scores = np.array(PRAUC_bootstrapped_scores)
    PRAUC_sorted_scores.sort()
    CI_PRAUC = [PRAUC_sorted_scores[int(0.025 * len(PRAUC_sorted_scores))], 
                PRAUC_sorted_scores[int(0.975 * len(PRAUC_sorted_scores))]]
    PRAUC = np.mean(PRAUC_sorted_scores)

    return ROCAUC, CI_ROCAUC, PRAUC, CI_PRAUC


def evaluate(selected_model, prediction_stacked, target_stacked, 
             hyperparamsearch, best_hyperparams, n_folds):
    
    """
    Stack the predictions and targets of all test months and evaluate these.
    The performance of the models on the test months is evaluated on the basis of 
    ROCAUC, PRAUC with their 95% confidence intervals, determined by bootstrapping.
    
    Parameters
    ----------
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    prediction_stacked: The predictions stacked for all test folds/months.
    target_stacked: The targets (true labels) stacked for all test folds/months.
    hyperparamsearch: Boolean. Whether the hyperparametersearch is conducted.
    best_hyperparams: List of the best hyperparameters as determined by the
        hyperparameter search.
    n_folds: Number of test folds/test months.
    
    Returns
    ---------
    ROCAUC: Area under the receiver operating curve.
    CI_ROCAUC: 95% confidence interval, determined by bootstrapping.
    PRAUC: Area under the precision recall curve.
    CI_PRAUC: 95% confidence interval, determined by bootstrapping.
    """

    if ( (prediction_stacked == None) or (target_stacked == None) ):
        print('No Evaluating')
        ROCAUC = None
        CI_ROCAUC = None
        PRAUC = None
        CI_PRAUC = None
    else:
        print('Start Evaluating')
            
        if ((selected_model=='LOGREG') or (selected_model=='SNN')):
            prediction_stacked = np.vstack(prediction_stacked)
        else:
            prediction_stacked = np.hstack(prediction_stacked)
        target_stacked = np.hstack(target_stacked)
        
        #evaluate prediction_stacked and target_stacked
        ROCAUC, CI_ROCAUC, PRAUC, CI_PRAUC = evaluate_test(prediction_stacked, target_stacked)
        
        print(f'ROC AUC test: {ROCAUC}')
        print(f'95% CI ROC AUC test: {CI_ROCAUC}')
        print(f'PR AUC test: {PRAUC}')
        print(f'95% CI PR AUC test: {CI_PRAUC}')
        
        if (hyperparamsearch):
            print(f'The best hyperparameters selected on the basis of the ROC AUC of ' +
                  f'the validation set for the {n_folds} test folds (outer loop) are: ' +
                  f'{best_hyperparams}')

    return ROCAUC, CI_ROCAUC, PRAUC, CI_PRAUC