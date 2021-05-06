#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at
        
For each model a hyperparameter search grid as well as the hyperparameter setting
as selected by prospective evaluation (experiment (iii) and (v)) are listed.

The hyperparameters for following models are listed:
    XGB: Extreme Gradient Boosting
    RF: Random Forest
    KNN: K-Nearest Neighbor
    SVM: Support Vector Machine
    LOGREG: Logistic Regression
    SNN: Self-Normalizing Neural Network

The boolean variable 'hyperparamsearch' is set in 'config.py'. This variable
determines whether to perform hyperparameter search or to use a fixed 
hyperparameter setting.
"""

from config import *

#search grid
if (hyperparamsearch):
    XGB_params = {
        'objective': ['binary:logistic'],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'eta': [0.1, 0.3, 0.6],
        'gamma': [0],
        'max_depth': [2,6,32],
        'scale_pos_weight': [True, False],
        'grow_policy': ['depthwise', 'lossguide'],
        'eval_metric': ['logloss'],
    }
#selected setting
else:
    XGB_params= {
        'objective': ['binary:logistic'],
        'booster': ['gbtree'],
        'eta': [0.1],
        'gamma': [0],
        'max_depth': [32],
        'scale_pos_weight': [True],
        'grow_policy': ['depthwise'],
        'eval_metric': ['logloss'],
    }

#search grid
if (hyperparamsearch):
    RF_params = {
        'n_estimators': [501],
        'criterion': ['gini', 'entropy'],
        'max_depth': [2,8,32,None],
        'min_samples_split': [2],
        'min_samples_leaf': [1,8,32],
        'max_features': ['auto', 'log2', None], #default 'auto'
        'max_leaf_nodes': [None],
        'class_weight': ['balanced', None],
    }
#selected setting
else:
    RF_params = {
        'n_estimators': [501],
        'criterion': ['gini'],
        'max_depth': [32],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['auto'], #default 'auto'
        'max_leaf_nodes': [None],  
        'class_weight': ['balanced'], 
    }

#search grid
if (hyperparamsearch):
    KNN_params = {
        'n_neighbors': [3,11,25,51,101,201,301],
        'weights': ['uniform', 'distance'],
    }
#selected setting
else:
    KNN_params = {
        'n_neighbors': [25],
        'weights': ['distance'],
    }

#search grid
if (hyperparamsearch):
    SVM_params = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'probability': [True],
        'class_weight': [None, 'balanced'],
    }
#selected setting
else:
    SVM_params = {
        'kernel': ['linear'],
        'probability': [True],
        'class_weight': [None],
    }

#search grid
if (hyperparamsearch):
    SVM_linear_params = {
        'dual': [False],
        'class_weight': [None, 'balanced']
    }
#selected setting
else:
    SVM_linear_params = {
        'dual': [False],
        'class_weight': [None]
    }

#search grid
if (hyperparamsearch):
    LOGREG_params = {
        'lr': [1e-2, 1e-3, 5e-4, 1e-4],
        'n_val_stops': [20],
        'weight_decay': [1e-5],
    }
#selected setting
else:
    LOGREG_params = {
        'lr': [1e-3],
        'n_val_stops': [20],
        'weight_decay': [1e-5],
    }

#search grid
if (hyperparamsearch):
    SNN_params = {
        'lr': [1e-3, 2e-4, 1e-4],
        'n_val_stops': [20],
        'weight_decay': [1e-5],
        'intermediate_size': [4, 16, 64],
        'n_layers': [1,3,6],
        'dropout': [0, 0.9],
    }
#selected setting
else:
    SNN_params = {
        'lr': [2e-4],
        'n_val_stops': [20],
        'weight_decay': [1e-5],
        'intermediate_size': [64],
        'n_layers': [3],
        'dropout': [0],
    }


