Project's modules and scripts description:

1. prep_train_data.py (inputDir | 'data/',
                       outputDir | 'inputs/') -> training_tweets.csv
    '''
    The script handles the datasets used : removes unnessesary columns, 
    changes lables, dropts short sentences. 
    On output it creates a csv file with training data.
    '''

2. datasets.py
    '''
    The module contains functions to create train/test datasets 
    and build the vocabulary.
    Used by training scripts.
    '''

3. utils.py
    '''
    The module with a class and a function to save the model's progress 
    and best performance.
    Also contains training and evaluation fucntions, metrics and time measurement.
    Used by training scripts.
    '''

4. model.py 
    '''
    The module contains model class.
    '''

5. init_train.py (inputDir | 'inputs/',
                  no_epochs,
                  outputDir | 'outputs/') -> model.pth, best_model.pth
    '''
    The script executes algorithm for model training.
    On output two files are created: 
        model.pth - stores information about the current state of the model,
        so that the training process can be resumed.
        best_model.pth - model's state at the best validation accuracy.
    '''

6. resume_train.py (pathToModel | 'outputs/',
                    inputDir | 'inputs/',
                    no_epochs) -> model.pth, best_model.pth
    '''
    The script executes the training algorithm from
    the point where it was stopped.
    Number of epochs is supposed to be the same as for the init_train.
    Overwrites the files created by inin_train.py
    '''