#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at

The required keys of the dataframe are:
    
obligatory keys:
    
patient_internal_id: identifies a patient. If a patient appears multiple times
    in the hospital, this id remains the same
    
case_internal_id: identifies a patient. If a patient shows up multiple times in 
    the hospital for different reasons, one patient gets multiple case_internal_id

Month: the month for prospective evaluation (for the 2019 set use an integer < 1)

corona: label, the result of the RT-PCR test

death_in_hospital: label, outcome, whether the patient died with COVID-19 or not

optional keys (at least 1 feature is obligatory):
    
type: hospital admission type (categorical: inpatient or outpatient)

gender: categorical

age: the patient age

... The remaining keys are blood test features. 
    these should correspond with the blood test features available at your 
    institution. You can add all possible blood test features here, the most 
    frequent features will be selected in the code.

"""

import numpy as np
import pandas as pd
import random

def generate_random_data(n_lab_entries, len_coronaset, data_path = 'data/'):
    
    '''
    Generates a random data set as a toy example and stores it to 
    'data_random.feather'
    
    Parameters
    ----------
    n_lab_entries: Random samples will be generated n_lab_entries times and 
        entered at a random feature and sample to random posititions.
    len_coronaset: The coronaset (the set of the data from 2020) have a length 
        len_coronaset. The samples from 2019 are stacked after the 2020 set.
    data_path: The path to the stored raw structure of the dataset and to where
        the generated data set will be stored at.
        
    Returns
    ---------
    True is return in case the method is successfully finished
    '''
    
    np.random.seed(0)
    
    path_data = str(data_path) + '/' + 'data_raw.feather'
    data_raw = pd.read_feather(path_data)
    
    path_statistics = str(data_path) + '/' + 'data_statistics.feather'
    data_statistics = pd.read_feather(path_statistics)
    
    mykeys = data_raw.keys()[8:]
    keylist = []
    for i in range(len(mykeys)):
        keylist.append(mykeys[i])
    random.shuffle(keylist)
    
    #generate death_in_hospital with COVID-19 label
    data_raw = generate_death_in_hospital_label(len_coronaset=len_coronaset, 
                                                data_raw=data_raw)
    
    #randomly generate lab data
    data_raw = generate_lab_entries(data_raw, n_lab_entries, keylist, 
                                    data_statistics)
    
    #add additional values for ferritin, neutrophils, calcium
    data_raw = add_additional_entries(data_raw, data_statistics)
    
    #modify the samples to intensify the connection to class
    data_raw = modify_samples(data_raw)
        
    #store file
    data_raw.to_feather(str(data_path)+'/'+'data_random.feather')
    
    return True


def generate_death_in_hospital_label(len_coronaset, data_raw):
    
    '''
    The 'death_in_hospital' label for a proportion of corona positive cases is 
    set to true 
    
    Parameters
    ----------
    len_coronaset: The length of the 2020 set. The 2019 set is stacked after 
        the 2020 set in the dataframe.
    data_raw: Pandas dataframe containing the data set with columns 
        'case_internal_id', 'corona' and 'death_in_hospital'.
        
    Returns
    ---------
    data_raw: Pandas data frame with added entries at 'death_in_hospital'.
    '''

    unique_ids = data_raw.loc[:len_coronaset, 'case_internal_id'].unique()
    for i, case_id in enumerate(unique_ids):
        indices = np.array(data_raw[data_raw['case_internal_id'] == case_id].index)
        if ((data_raw.iloc[indices]['corona']==1).any()):
            data_raw.loc[indices,'death_in_hospital'] = \
                np.random.choice(np.arange(0, 2), p=[0.95, 0.05])
        else:
            data_raw.loc[indices,'death_in_hospital'] = 0.0
    
    return data_raw


def generate_lab_entries(data_raw, n_lab_entries, keylist, data_statistics):
    
    '''
    Generate blood test entries by randomly selecting the feature and the row.
    The feature value is sampled from a Gaussian with a mean and standard deviation
    calculated from the original data distribution from the corona positive and 
    negative class.
    
    Parameters
    ----------
    data_raw: Pandas dataframe containing the data set.
    n_lab_entries: Random samples will be generated n_lab_entries times and 
        entered at a random feature and sample to random posititions.
    keylist: List of the keys (strings) at which random samples are added to.
    data_statistics: Contains the mean and standard deviation of the COVID-19
        positive and negative class for each key. 
        
    Returns
    ---------
    data_raw: Pandas data frame with added random features (sampled from a 
        Gaussian distribution).
    '''
    
    for i in range(n_lab_entries):
        #select a key
        key = int(np.random.sample(1)*len(keylist))
        #select a row
        row = np.random.randint(0, len(data_raw))
        #write random value to entry
        if (data_raw.loc[row,'corona']==1):
            data_raw.loc[row,keylist[key]] = ((np.random.randn()*
                    data_statistics.iloc[key]['cov_pos_sd'])+
                    data_statistics.iloc[key]['cov_pos_mean'])
        else:
            data_raw.loc[row,keylist[key]] = ((np.random.randn()*
                    data_statistics.iloc[key]['cov_neg_sd'])+
                    data_statistics.iloc[key]['cov_neg_mean'])
    
    return data_raw

def add_additional_entries(data_raw, data_statistics):
    
    '''
    Additional entries are added for Ferritin, Neutropiles and Calcium to ensure
    that these features to ensure selection of these features.
    
    Parameters
    ----------
    data_raw: Pandas dataframe containing the data set.
    data_statistics: Contains the mean and standard deviation of the COVID-19
        positive and negative class for each key. 
        
    Returns
    ---------
    data_raw: Pandas data frame with additional random features.
    '''
    
    key_FER = np.where(data_raw.keys()=='FER')[0]-8
    key_neutro = np.where(data_raw.keys()=='_NEUTROPHILE')[0]-8
    key_calcium = np.where(data_raw.keys()=='_CALCIUM')[0]-8
    for i in range(int(len(data_raw)*0.25)):
        #randomly select a row
        row = np.random.randint(0, len(data_raw))
        
        #write random value to entry
        if (data_raw.loc[row,'corona']==1):
            data_raw.loc[row,'FER'] = ((np.random.randn()*
                    data_statistics.iloc[key_FER]['cov_pos_sd'].values)+
                    data_statistics.iloc[key_FER]['cov_pos_mean'].values)
            data_raw.loc[row,'_NEUTROPHILE'] = ((np.random.randn()*
                    data_statistics.iloc[key_neutro]['cov_pos_sd'].values)+
                    data_statistics.iloc[key_neutro]['cov_pos_mean'].values)
            data_raw.loc[row,'_CALCIUM'] = ((np.random.randn()*
                    data_statistics.iloc[key_calcium]['cov_pos_sd'].values)+
                    data_statistics.iloc[key_calcium]['cov_pos_mean'].values)
        else:
            data_raw.loc[row,'FER'] = ((np.random.randn()*
                    data_statistics.iloc[key_FER]['cov_neg_sd'].values)+
                    data_statistics.iloc[key_FER]['cov_neg_mean'].values)
            data_raw.loc[row,'_NEUTROPHILE'] = ((np.random.randn()*
                    data_statistics.iloc[key_neutro]['cov_neg_sd'].values)+
                    data_statistics.iloc[key_neutro]['cov_neg_mean'].values) 
            data_raw.loc[row,'_CALCIUM'] = ((np.random.randn()*
                    data_statistics.iloc[key_calcium]['cov_neg_sd'].values)+
                    data_statistics.iloc[key_calcium]['cov_neg_mean'].values )
     
    return data_raw

def modify_samples(data_raw):
    
    '''
    A corona class specific offset is added to Ferritin and Calcium, and for the
    death_in_hospital class to Neutrophiles. This allows the models to train from
    the randomly generated (toy example) data set.
    
    Parameters
    ----------
    data_raw: Pandas dataframe containing the data set.
    
    Returns
    ---------
    data_raw: Pandas data frame with modified feature values to connect them 
        with a class.
    '''

    # elevate Ferritin for corona positive samples
    data_raw.loc[data_raw['corona']==1,'FER']+=500
    data_raw.loc[data_raw['corona']==0,'FER']-=500
    # lower Calcium for corona positive samples
    data_raw.loc[data_raw['corona']==1,'_CALCIUM']-=0.5
    data_raw.loc[data_raw['corona']==0,'_CALCIUM']+=0.5
    
    # elevate Neutrophils for death_in_hospital positive samples
    data_raw.loc[data_raw['death_in_hospital']==1,'_NEUTROPHILE']+=20
    data_raw.loc[data_raw['death_in_hospital']==0,'_NEUTROPHILE']-=20
    
    return data_raw