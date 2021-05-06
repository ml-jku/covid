'''
Author: Theresa Roland
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz

Contact: roland@ml.jku.at       

Set the configurations in this file.
'''


''' ---------------------------------------------------------------------------
Generation of random data.
In case you use your own data, see 'generate_random_data.py' 
for more information about the required structure of the data set.
--------------------------------------------------------------------------- ''' 

''' The dataset is already generated and stored in 
    ../data/data_random.feather '''
do_generate_random_data = False

''' Path to store the random generated data '''
data_path = '../data/'

''' Number of lab entries '''
n_lab_entries = 15000000

''' Number of samples in the 2020 set (number of tested samples) '''
len_coronaset = 11290


''' ---------------------------------------------------------------------------
General settings
--------------------------------------------------------------------------- ''' 

''' Conduct hyperparameter search (hyperparameters are listed in 
    'hyperparams.py') '''
hyperparamsearch = False

''' Select a minimum of n most frequent features. 
    The model will be trained on the basis of these features. '''
min_n_features = 100

''' Each samples requires a minimum of 'min_n_entries' measured values.
    Samples with less than 'min_n_entries' are discarded. '''
min_n_entries = 8

''' Select label. COVID-19 diagnosis prediction = corona
    COVID-19 associated mortality risk = death_in_hospital
    options: 'corona', 'death_in_hospital' '''
label_column = 'corona'

''' Select a model architecture from following options:
    Self-Normalizing Neural Network: 'SNN'
    Logistic Regression:  'LOGREG'
    Random Forest: 'RF'
    Extreme Gradient Boosting: 'XGB'
    K-Nearest Neighbor: 'KNN'
    Support Vector Machine: 'SVM' '''
selected_model = 'XGB' 

''' True, if the negatives dataset (the dataset from 2019) is used. '''
use_negatives_dataset = True 

''' The month your trained model will be tested on. '''
test_months = [11, 12]

''' Whether you want to store the trained model. The model will only be stored
    if the test_months have a maximum length of 1, otherwise more than 1 model
    is trained. '''
store_trained_model = False

''' The path to store the model at. '''
path_store_model = 'trained_model/'

''' For CPU use: 'cpu', for GPU use e.g.: 'cuda:0' '''
device = 'cuda:0'


''' ---------------------------------------------------------------------------
Recent samples are weighted higher than older samples to counter the domain 
shift. Samples from 2019 are weighted with 0.01 instead of the default of 1.
--------------------------------------------------------------------------- ''' 

''' The last entry is the weight of the last training month, the second last
    entry is the weight of the second last training month, ... '''
sampling_2020_weights = [1,1,2,3]

''' The weight of the 2019 samples '''
sampling_2019_weights = 0.01


''' ---------------------------------------------------------------------------
Training
--------------------------------------------------------------------------- ''' 

''' Number of validation folds per hyperparameter setting '''
n_val_folds = 1

''' Batch size for Logistic Regression and Self-Normalizing Neural Network '''
batch_size = 256

''' Criterion to calculate Loss between prediction and target for Logistic 
    Regression and Self-Normalizing Neural Network '''
criterion = 'BCEWithLogitsLoss'