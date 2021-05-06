#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at
        
Store and load trained models
"""

import torch
import pickle

def store_model(store_trained_model, selected_model, label_column, final_model,
                test_months = [], path_store_model = 'trained_model/'):
    
    '''
    Store a trained model. The model is only stored, if 'store_trained_model' is
    True and test_months has no more than 1 entry.
    
    Parameters
    ----------
    store_trained_model: Boolean, whether to store the trained model.
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    label_column: Indicating what is the task. COVID_19 diagnosis prediction 
        with 'corona' or mortality prediciton with 'death_in_hospital'. 
    final_model: Trained torch model (e.g., Logistic Regression model or 
        Self-Normalizing Neural Network model).
    test_months: A list containing the integers of the test months (the months, 
        the model is tested on).
    path_store_model: Where to store the model to.
    
    Returns
    ---------
    stored: Whether the model was stored.
    '''
    
    if (store_trained_model and (len(test_months)<1)):
        filename = 'model_'  + str(selected_model) + '_' + str(label_column) + '.sav'
        path = path_store_model + filename
        if (selected_model in ['SNN', 'LOGREG']):            
            torch.save(final_model, open(path, 'wb'))
        else:
            pickle.dump(final_model, open(path, 'wb'))
        print(f'Trained model is stored: {path}')
        stored = True
    else: 
        stored = False

    return stored


def load_model(selected_model, path_load_model):
    
    '''
    Load a stored model.
    
    Parameters
    ----------
    selected_model: A string of the selected model. E.g., 'LOGREG' for Logistic 
        Regression. See 'config.py' for the options.
    path_load_model: Where to model to load is located. Including the name and 
        file ending of the model.
    
    Returns
    ---------
    model: The loaded model.
    '''
    
    if (selected_model in ['SNN', 'LOGREG']):            
        model =  torch.load(open(path_load_model, 'rb'))
    else:
        model = pickle.load(open(path_load_model, 'rb'))

    print('Model is loaded')

    return model
