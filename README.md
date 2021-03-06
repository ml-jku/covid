# COVID-19 Diagnosis from Blood Tests with Robustness to Domain Shifts
Theresa Roland, Carl Böck, Thomas Tschoellitsch, Alexander Maletzky, Sepp Hochreiter, Jens Meier, Günter Klambauer

ABSTRACT. We investigate machine learning models that identify COVID-19   positive   patients   and   estimate   the   mortality   risk based  on  routinely  acquired  blood  tests  in  a  hospital  setting. However, during pandemics or new outbreaks, disease and testing characteristics   change,   thus   we   face   domain   shifts.   Domain shifts  can  be  caused,  e.g.,  by  changes  in  the  disease  prevalence (spreading  or  tested  population),  by  refined  RT-PCR  testing procedures  (taking  samples,  laboratory),  or  by  virus  mutations. Therefore, machine learning models for diagnosing COVID-19 or other  diseases  may  not  be  reliable  and  degrade  in  performance over  time.  To  counter  this  effect,  we  propose  methods  that  first identify domain shifts and then reverse their negative effects on the model performance. Frequent re-training and re-assessment, as  well  as  stronger  weighting  of  more  recent  samples,  keeps model  performance  and  credibility  at  a  high  level  over  time. Our  diagnosis  models  are  constructed  and  tested  on  large-scale data sets, steadily adapt to observed domain shifts, and maintain high  ROC  AUC  values  along  pandemics.

Link to [Preprint](https://www.medrxiv.org/content/10.1101/2021.04.06.21254997v1.full-text).

## Code
We provide code to train a machine learning model for COVID-19 diagnosis or mortality prediction for your institution.

Note that we are currently in the process of uploading and cleaning up code. There might be some bugs which will be fixed soon. If you need help with running the example in the meantime, feel free to write me an email (roland at ml.jku.at).

### Setup
For running experiments or training the models, you need Python 3.8.2 or later.
```
$ conda install python==3.8.2
```
Make sure to satisfy the requirements as listed in the file ```requirements.txt```:
```
$ pip install -r requirements.txt
```
Change the current directory to the example folder:
```
$ cd PATH/example/
```
After adapting the configurations in ```config.py``` and the hyperparameters in ```hyperparams.py```, the example script can be run:
```
$ python example.py
```

### Data set
For ethical reasons, the data set described in the paper is not publicly available. We provide a randomly generated data set (toy example) to test the code and to 
demonstrate the required structure of the data set. The data set can be generated by setting ```do_generate_random_data = True``` in the file ```config.py```. Note that the generation of the random data set is slow, however, we uploaded the generated data set: ```data/data_random.feather```.

IMPORTANT: Do not use models trained on the toy example data set for real-world applications as this data set is randomly generated!

Replace the path to the toy example data set with the path to your data set in the file ```config.py```: ```data_path = PATH_TO_DATA```. Ensure that the data set has the required structure as described in the header of file ```generate_random_data.py```.

### Hyperparameter search
The code allows grid search to identify suitable hyperparameters. If you want to perform hyperparameter search, set 
```hyperparamsearch = True```
in the file ```config.py```. You can define the hyperparameter search grid in the file ```hyperparams.py```. In case of no hyperparametersearch set the variable ```hyperparamsearch = False``` and adapt the hyperparameters for your model in the file ```hyperparams.py```.
The hyperparameter search will be conducted on the basis of a validation set. The best hyperparameter setting/s on the validation set/s will be printed.

### Prospective evaluation
For the assessement of the predictive performance, we suggest prospective evaluation, rather than random cross-validation. You can set the months you want to assess your model on in the file ```config.py``` in the list ```test_months```. In the paper, we suggest to weight recent samples higher than older samples, this is implemented in the code and the weighting can be adapted in the file ```config.py``` at the variables ```sampling_2019_weights``` and ```sampling_2020_weights```.

### Train model on all available data
For real-world application, we suggest to train the model on all available data. The model will be stored to the specified path. Therefore, ensure following settings in the file ```config.py```:
```
hyperparametersearch = False
test_months = []
store_trained_model = True
path_store_model = PATH_TO_STORE_TRAINED_MODEL
```
